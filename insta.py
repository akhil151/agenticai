import asyncio
import logging
import os
import io
import glob
from huggingface_hub import InferenceClient
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
from groq import Groq

# Load .env
load_dotenv()

# ── Directories ────────────────────────────────────────────────────────────────
IMAGE_CACHE_DIR = "./image_cache"
STATIC_DIR      = "./static"
SCREENSHOTS_DIR = "./screenshots/instagram"
LOGS_DIR        = "./logs"

for _d in [IMAGE_CACHE_DIR, STATIC_DIR, SCREENSHOTS_DIR, LOGS_DIR, "./insta_session"]:
    os.makedirs(_d, exist_ok=True)

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"{LOGS_DIR}/instagram_automation.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
GROQ_MODEL    = "llama-3.3-70b-versatile"


# ══════════════════════════════════════════════════════════════════════════════
# Image helpers
# ══════════════════════════════════════════════════════════════════════════════

def generate_image_with_huggingface(prompt: str) -> str:
    """Generate image locally. Returns local path or None on failure."""
    from local_image_generator import generate_local_image
    path = generate_local_image(prompt, IMAGE_CACHE_DIR, lambda: None)
    if path:
        cleanup_cache()
    return path


def cleanup_cache(max_files: int = 100):
    files = sorted(glob.glob(os.path.join(IMAGE_CACHE_DIR, "*.png")), key=os.path.getctime)
    for old in files[:-max_files]:
        os.remove(old)


# ══════════════════════════════════════════════════════════════════════════════
# Text helper
# ══════════════════════════════════════════════════════════════════════════════

def generate_post_text(client: Groq, domain_name: str, custom_prompt: str, day: int) -> str:
    prompt = f"""
Generate a short, engaging Instagram caption about "{custom_prompt}" in the context of "{domain_name}".
Make it exciting, add relevant emojis and hashtags, and keep it under 150 words.
Include "#Day{day}" at the end to make it part of a series.
"""
    try:
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates engaging social media captions."},
                {"role": "user",   "content": prompt}
            ],
            max_tokens=300,
            temperature=0.7
        )
        text = resp.choices[0].message.content.strip()
        logger.info(f"Generated text: {text[:80]}...")
        return text
    except Exception as e:
        logger.error(f"Groq error: {e}")
        return f"🚀 Default Instagram caption about {custom_prompt} (via {domain_name}) #Day{day}"


# ══════════════════════════════════════════════════════════════════════════════
# Login wait
# ══════════════════════════════════════════════════════════════════════════════

async def wait_for_login(page):
    """Poll every 3 s until Instagram home is accessible."""
    logger.info("🔑 Instagram login required.")
    logger.info("👉 Please log in manually in the browser — the script continues automatically once done...")
    while True:
        try:
            url = page.url

            # If Instagram shows popups after login, close them safely.
            for txt in ["Not Now", "Not now", "Cancel"]:
                try:
                    popup_btn = page.get_by_text(txt, exact=True)
                    if await popup_btn.count() > 0 and await popup_btn.first().is_visible():
                        await popup_btn.first().click(force=True)
                        logger.info(f"✅ Closed popup: {txt}")
                        await page.wait_for_timeout(1000)
                except Exception:
                    pass

            on_login = (
                "login"      in url.lower() or
                "challenge"  in url.lower() or
                "checkpoint" in url.lower() or
                await page.query_selector("input[name='username']") is not None
            )
            if not on_login:
                try:
                    await page.wait_for_selector(
                        "a[href='/'], svg[aria-label='Home'], svg[aria-label='Create'], svg[aria-label='New post'], a[href='/explore/'], a[href='/direct/inbox/']",
                        timeout=5000
                    )
                    logger.info("✅ Login confirmed — continuing...")
                    return
                except PlaywrightTimeoutError:
                    pass
        except Exception:
            pass
        await asyncio.sleep(3)


# ══════════════════════════════════════════════════════════════════════════════
# Posting
# ══════════════════════════════════════════════════════════════════════════════

async def post_to_instagram(page, post_text: str, image_path: str = None):
    ss = lambda name: f"{SCREENSHOTS_DIR}/{name}.png"
    try:
        if not image_path or not os.path.exists(image_path):
            logger.error("❌ Instagram needs an image/video file. No valid image found.")
            await page.screenshot(path=ss("image_missing"))
            return False

        # ── Open composer ──────────────────────────────────────────────────
        logger.info("📝 Opening Instagram create composer...")
        editor_button = None

        # First try normal Instagram create buttons.
        for sel in [
            "svg[aria-label='New post']",
            "svg[aria-label='Create']",
            "a[href='/create/select/']",
            "div[role='button']:has-text('Create')",
            "span:has-text('Create')"
        ]:
            try:
                editor_button = await page.wait_for_selector(sel, timeout=8000, state="visible")
                if editor_button:
                    await editor_button.click(force=True)
                    logger.info(f"✅ Composer button clicked via: {sel}")
                    try:
                        await page.wait_for_timeout(1500)
                        loc = page.get_by_text("Post", exact=True)
                        if await loc.count() > 0:
                            for i in range(await loc.count()):
                                btn = loc.nth(i)
                                if await btn.is_visible():
                                    await btn.click(force=True)
                                    logger.info("✅ Clicked 'Post' from the Create sub-menu.")
                                    break
                    except Exception as e:
                        logger.debug(f"Sub-menu 'Post' click failed/skipped: {e}")
                    break
            except PlaywrightTimeoutError:
                continue
            except Exception:
                continue

        # Fallback: direct create URL.
        if not editor_button:
            logger.warning("⚠️ Could not click Create button — opening create URL directly")
            await page.goto("https://www.instagram.com/create/select/", wait_until="domcontentloaded", timeout=60000)
            await page.wait_for_timeout(3000)

        await page.wait_for_timeout(2000)
        await page.screenshot(path=ss("create_opened"))

        # ── Image upload ───────────────────────────────────────────────────
        uploaded = False
        logger.info(f"📸 Uploading image: {image_path}")
        await page.screenshot(path=ss("before_image_upload"))

        for retry in range(3):
            try:
                for sel in [
                    "input[type='file'][accept*='image']",
                    "input[type='file'][accept*='video']",
                    "input[type='file']"
                ]:
                    try:
                        fi = await page.wait_for_selector(sel, state="attached", timeout=10000)
                        await fi.set_input_files(os.path.abspath(image_path))
                        logger.info(f"✅ Image via direct input (retry {retry+1})")
                        uploaded = True
                        break
                    except PlaywrightTimeoutError:
                        continue
                if uploaded:
                    break
                await page.wait_for_timeout(2000)
            except Exception as re_e:
                logger.debug(f"Upload retry {retry+1}: {re_e}")

        if uploaded:
            try:
                await page.wait_for_selector(
                    "img[alt='Photo by'], canvas, div[role='dialog'] img, div[role='dialog'] canvas",
                    timeout=15000
                )
                logger.info("✅ Image preview confirmed")
            except PlaywrightTimeoutError:
                logger.warning("⚠️ Image preview not detected")
            await page.wait_for_timeout(4000)
            await page.screenshot(path=ss("after_image_upload"))
        else:
            logger.error("❌ All upload retries failed")
            await page.screenshot(path=ss("image_upload_failed"))
            return False

        # ── Click Next buttons ─────────────────────────────────────────────
        logger.info("➡️ Clicking Next button...")
        await page.screenshot(path=ss("before_first_next"))
        next_success = await click_by_text_or_role(page, ["Next"], timeout_ms=10000)
        if not next_success:
            logger.error("❌ Failed to click first Next button")
            await page.screenshot(path=ss("first_next_failed"))
            return False

        await page.wait_for_timeout(3000)

        # Instagram often has a second Next screen for crop/filter/edit.
        logger.info("➡️ Clicking second Next button if available...")
        await click_by_text_or_role(page, ["Next"], timeout_ms=5000, required=False)
        await page.wait_for_timeout(3000)
        await page.screenshot(path=ss("caption_screen"))

        # ── Fill text ──────────────────────────────────────────────────────
        logger.info("⌨ Filling caption editor...")
        textbox = None
        for sel in [
            "div[aria-label='Write a caption...']",
            "textarea[aria-label='Write a caption...']",
            "div[role='textbox']",
            "textarea"
        ]:
            try:
                textbox = await page.wait_for_selector(sel, timeout=10000, state="visible")
                if textbox:
                    break
            except PlaywrightTimeoutError:
                continue

        if not textbox:
            logger.error("❌ Could not find caption textbox")
            await page.screenshot(path=ss("text_box_not_found"))
            return False

        await textbox.click()
        await page.wait_for_timeout(300)

        try:
            await textbox.fill(post_text)
            logger.info("✅ Caption filled via fill()")
        except Exception:
            try:
                await page.keyboard.insert_text(post_text)
                logger.info("✅ Caption inserted via keyboard fallback")
            except Exception as e2:
                logger.error(f"❌ Could not fill caption: {e2}")
                await page.screenshot(path=ss("text_fill_failed"))
                return False

        await page.wait_for_timeout(1000)

        # ── Click Share button ─────────────────────────────────────────────
        logger.info("🚀 Clicking Share button...")
        await page.screenshot(path=ss("before_post_click"))
        post_success = await click_by_text_or_role(page, ["Share"], timeout_ms=15000)

        if not post_success:
            logger.error("❌ Failed to click Share button")
            await page.screenshot(path=ss("post_click_failed"))
            return False

        try:
            await page.wait_for_selector(
                "text=Your post has been shared, text=Your reel has been shared, text=Post shared",
                timeout=30000
            )
            logger.info("✅ Post published successfully!")
        except PlaywrightTimeoutError:
            logger.warning("⚠️ Success message not detected — verify manually in Instagram")

        await page.wait_for_timeout(5000)
        await page.screenshot(path=ss("post_success"))
        return True

    except Exception as e:
        logger.error(f"❌ Error during posting: {e}")
        await page.screenshot(path=ss("posting_error"))
        return False


async def click_by_text_or_role(page, names, timeout_ms=8000, required=True):
    """Small helper, kept separate so post_to_instagram stays similar to fb.py."""
    deadline = datetime.now().timestamp() + (timeout_ms / 1000)

    while datetime.now().timestamp() < deadline:
        for exact_name in names:
            # Role-based click first.
            try:
                locators_to_try = [
                    page.get_by_role("button", name=exact_name, exact=True),
                    page.get_by_text(exact_name, exact=True)
                ]
                for loc in locators_to_try:
                    if await loc.count() > 0:
                        for i in range(await loc.count()):
                            btn = loc.nth(i)
                            if await btn.is_visible():
                                await btn.scroll_into_view_if_needed()
                                await btn.click(force=True)
                                logger.info(f"✅ Button clicked: {exact_name}")
                                return True
            except Exception:
                pass

            # Selector fallback.
            for sel in [
                f"div[role='button']:text-is('{exact_name}')",
                f"button:text-is('{exact_name}')",
                f"span:text-is('{exact_name}')",
                f"div:has-text('{exact_name}')"
            ]:
                try:
                    locator = page.locator(sel)
                    count = await locator.count()
                    for idx in range(count):
                        try:
                            btn = locator.nth(idx)
                            await btn.wait_for(state="visible", timeout=1000)
                            text = await btn.inner_text()
                            if text.strip() != exact_name:
                                continue
                            await btn.scroll_into_view_if_needed()
                            await btn.click(force=True)
                            logger.info(f"✅ Button clicked: {sel} [index {idx}]")
                            return True
                        except Exception:
                            continue
                except Exception:
                    continue

        await page.wait_for_timeout(500)

    if required:
        logger.error(f"❌ Required button not found: {names}")
    return False


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

async def run():
    # ── Collect inputs ─────────────────────────────────────────────────────
    domain_name = input("Domain Name: ").strip()
    if not domain_name:
        logger.error("Domain Name is required.")
        return

    custom_prompt = input("Custom prompt (press Enter to skip): ").strip() or "🚀 Automated Instagram Post!"

    want_image = input("Do you want an image? (y/n): ").strip().lower() == "y"
    image_prompt = ""
    if want_image:
        image_prompt = input("Image prompt: ").strip()
        if not image_prompt:
            logger.error("Image prompt is required when image is enabled.")
            return

    num_days_raw = input("Number of days (default 1): ").strip()
    num_days = int(num_days_raw) if num_days_raw.isdigit() else 1

    # ── Groq client ────────────────────────────────────────────────────────
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        logger.error("❌ GROQ_API_KEY not found in .env — please add it and retry.")
        return
    logger.info("✅ Groq API key loaded from .env")
    client = Groq(api_key=groq_api_key)

    # ── Browser ────────────────────────────────────────────────────────────
    async with async_playwright() as p:
        headless = os.getenv("HEADLESS", "false").lower() == "true"
        browser  = await p.chromium.launch_persistent_context(
            user_data_dir="./insta_session",
            headless=headless,
            args=["--start-maximized"] if not headless else []
        )
        page = await browser.new_page()

        try:
            logger.info("🌐 Navigating to Instagram...")
            await page.goto("https://www.instagram.com", wait_until="domcontentloaded", timeout=60000)
            await page.wait_for_timeout(3000)

            # Login check
            is_login = (
                "login" in page.url.lower() or
                "challenge" in page.url.lower() or
                "checkpoint" in page.url.lower() or
                await page.query_selector("input[name='username']") is not None
            )
            if is_login:
                await wait_for_login(page)
            else:
                logger.info("✅ Already logged in via saved session.")

            # Post loop
            for day in range(num_days):
                logger.info(f"📅 Day {day + 1}/{num_days}")
                post_text  = generate_post_text(client, domain_name, custom_prompt, day + 1)
                image_path = generate_image_with_huggingface(image_prompt) if want_image else None
                success    = await post_to_instagram(page, post_text, image_path)

                if success:
                    logger.info(f"✅ Day {day + 1} posted successfully")
                else:
                    logger.error(f"❌ Day {day + 1} post failed")

                if day < num_days - 1:
                    logger.info("⏳ Pausing before next post...")
                    await page.wait_for_timeout(5000)

        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            await page.screenshot(path=f"{SCREENSHOTS_DIR}/unexpected_error.png")
        finally:
            await page.wait_for_timeout(5000)
            await browser.close()


if __name__ == "__main__":
    asyncio.run(run())
