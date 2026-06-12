import asyncio
import logging
import os
import glob
from huggingface_hub import InferenceClient
from datetime import datetime
from PIL import Image
from dotenv import load_dotenv
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
from groq import Groq

# Load .env
load_dotenv()

# ── Directories ────────────────────────────────────────────────────────────────
IMAGE_CACHE_DIR = "./image_cache"
STATIC_DIR      = "./static"
SCREENSHOTS_DIR = "./screenshots/linkedin"
LOGS_DIR        = "./logs"

for _d in [IMAGE_CACHE_DIR, STATIC_DIR, SCREENSHOTS_DIR, LOGS_DIR, "./linkedin_session"]:
    os.makedirs(_d, exist_ok=True)

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"{LOGS_DIR}/linkedin_automation.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
GROQ_MODEL = "llama-3.3-70b-versatile"


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
Generate a professional, engaging LinkedIn post about "{custom_prompt}" in the context of "{domain_name}".
Make it valuable, clear, and human-written. Use a confident but not overly casual tone.
Add 3 to 6 relevant hashtags, keep it under 180 words, and include "#Day{day}" at the end to make it part of a series.
"""
    try:
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates professional LinkedIn posts."},
                {"role": "user",   "content": prompt}
            ],
            max_tokens=350,
            temperature=0.7
        )
        text = resp.choices[0].message.content.strip()
        logger.info(f"Generated text: {text[:80]}...")
        return text
    except Exception as e:
        logger.error(f"Groq error: {e}")
        return f"🚀 Default LinkedIn post about {custom_prompt} in {domain_name}. #Day{day}"


# ══════════════════════════════════════════════════════════════════════════════
# Login wait
# ══════════════════════════════════════════════════════════════════════════════

async def wait_for_login(page):
    """Poll every 3 s until LinkedIn feed is accessible after manual login."""
    logger.info("🔑 LinkedIn login required.")
    
    email = os.getenv("LINKEDIN_EMAIL")
    password = os.getenv("LINKEDIN_PASSWORD")
    if email and password:
        logger.info("🤖 Attempting automatic LinkedIn login...")
        try:
            user_loc = page.locator("input[name='session_key'], input#username")
            if await user_loc.count() > 0 and await user_loc.first.is_visible():
                await user_loc.first.fill(email)
                
                pass_loc = page.locator("input[name='session_password'], input#password")
                if await pass_loc.count() > 0:
                    await pass_loc.first.fill(password)
                
                sign_in_btn = page.locator("button[type='submit'], button[aria-label='Sign in']")
                if await sign_in_btn.count() > 0:
                    await sign_in_btn.first.click()
                    logger.info("✅ Credentials submitted! Waiting 5s to see if login succeeds...")
                    await page.wait_for_timeout(5000)
        except Exception as e:
            logger.warning(f"⚠️ Auto-login attempt failed: {e}")

    logger.info("👉 Please complete login manually if needed (e.g. CAPTCHA) — the script continues automatically once done...")
    while True:
        try:
            url = page.url.lower()

            # Close safe popups if LinkedIn shows them after login.
            for txt in ["Dismiss", "Skip", "Not now", "Maybe later", "No thanks", "Got it"]:
                try:
                    popup_btn = page.get_by_text(txt, exact=True)
                    if await popup_btn.count() > 0 and await popup_btn.first().is_visible():
                        await popup_btn.first().click(force=True)
                        logger.info(f"✅ Closed popup: {txt}")
                        await page.wait_for_timeout(1000)
                except Exception:
                    pass

            on_login = (
                "login" in url or
                "checkpoint" in url or
                "challenge" in url or
                "signup" in url or
                await page.query_selector("input[name='session_key'], input#username") is not None
            )

            if not on_login:
                try:
                    await page.wait_for_selector(
                        "button.share-box-feed-entry__trigger, button[aria-label*='Start a post'], a[href*='/feed/'], div.feed-identity-module",
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

async def post_to_linkedin(page, post_text: str, image_path: str = None):
    ss = lambda name: f"{SCREENSHOTS_DIR}/{name}.png"
    try:
        if image_path and not os.path.exists(image_path):
            logger.error("❌ Image path was provided, but file does not exist.")
            await page.screenshot(path=ss("image_missing"))
            return False

        # ── Open composer ──────────────────────────────────────────────────
        logger.info("📝 Opening LinkedIn post composer...")
        await page.goto("https://www.linkedin.com/feed/", wait_until="domcontentloaded", timeout=60000)
        await page.wait_for_timeout(3000)
        await page.screenshot(path=ss("feed_opened"))

        editor_button = None
        for sel in [
            "text='Start a post'",
            "button.share-box-feed-entry__trigger",
            "button[aria-label*='Start a post']",
            "button:has-text('Start a post')",
            "div.share-box-feed-entry__top-bar button",
            "span:has-text('Start a post')"
        ]:
            try:
                editor_button = await page.wait_for_selector(sel, timeout=8000, state="visible")
                if editor_button:
                    await editor_button.click(force=True)
                    logger.info(f"✅ Composer button clicked via: {sel}")
                    break
            except PlaywrightTimeoutError:
                continue
            except Exception:
                continue

        if not editor_button:
            logger.error("❌ Could not open LinkedIn post composer")
            await page.screenshot(path=ss("composer_open_failed"))
            return False

        await page.wait_for_timeout(2000)
        await page.screenshot(path=ss("composer_opened"))

        # ── Fill text ──────────────────────────────────────────────────────
        logger.info("⌨ Filling LinkedIn post editor...")
        textbox = None
        for sel in [
            "div.ql-editor[contenteditable='true']",
            "div[role='textbox'][contenteditable='true']",
            "div[aria-label*='Text editor']",
            "div[aria-label*='What do you want to talk about']",
            "div[contenteditable='true']"
        ]:
            try:
                textbox = await page.wait_for_selector(sel, timeout=12000, state="visible")
                if textbox:
                    break
            except PlaywrightTimeoutError:
                continue

        if not textbox:
            logger.error("❌ Could not find LinkedIn text editor")
            await page.screenshot(path=ss("text_box_not_found"))
            return False

        await textbox.click()
        await page.wait_for_timeout(300)

        try:
            await textbox.fill(post_text)
            logger.info("✅ Post text filled via fill()")
        except Exception:
            try:
                await page.keyboard.insert_text(post_text)
                logger.info("✅ Post text inserted via keyboard fallback")
            except Exception as e2:
                logger.error(f"❌ Could not fill LinkedIn post text: {e2}")
                await page.screenshot(path=ss("text_fill_failed"))
                return False

        await page.wait_for_timeout(1000)
        await page.screenshot(path=ss("after_text_fill"))

        # ── Optional image upload ──────────────────────────────────────────
        if image_path:
            uploaded = await upload_linkedin_image(page, image_path, ss)
            if not uploaded:
                return False
        else:
            logger.info("ℹ️ No image selected — posting text-only LinkedIn post.")

        # ── Click Post button ──────────────────────────────────────────────
        logger.info("🚀 Clicking LinkedIn Post button...")
        await page.screenshot(path=ss("before_post_click"))
        
        post_success = False
        start_time = datetime.now().timestamp()
        
        while datetime.now().timestamp() - start_time < 30 and not post_success:
            for sel in [
                "button.share-actions__primary-action",
                "div.share-box_actions button.primary-action",
                "button:has-text('Post'):not(:has-text('Start a post'))",
                "button:has(span:text-is('Post'))",
                "button[aria-label='Post']",
                "div[role='button']:has-text('Post')",
                "span:text-is('Post')"
            ]:
                if post_success:
                    break
                try:
                    elements = page.locator(sel)
                    for i in range(await elements.count()):
                        btn = elements.nth(i)
                        if await btn.is_visible():
                            disabled_attr = await btn.get_attribute("disabled")
                            aria_disabled = await btn.get_attribute("aria-disabled")
                            
                            is_disabled_pw = False
                            try:
                                is_disabled_pw = await btn.is_disabled()
                            except:
                                pass
                                
                            if is_disabled_pw or (disabled_attr is not None) or (aria_disabled and aria_disabled.lower() == "true"):
                                continue # Visible but disabled, skip and retry later
                                
                            await btn.scroll_into_view_if_needed()
                            await btn.click(force=True)
                            logger.info(f"✅ Post button clicked via: {sel}")
                            post_success = True
                            break
                except Exception:
                    pass
            if not post_success:
                await page.wait_for_timeout(2000)

        # Fallback if the specific selectors failed
        if not post_success:
            post_success = await click_by_text_or_role(page, ["Post"], timeout_ms=5000)

        if not post_success:
            logger.error("❌ Failed to click LinkedIn Post button")
            await page.screenshot(path=ss("post_click_failed"))
            return False

        try:
            await page.wait_for_selector(
                "text=Post successful, text=Your post has been shared, text=View post",
                timeout=30000
            )
            logger.info("✅ LinkedIn post published successfully!")
        except PlaywrightTimeoutError:
            logger.warning("⚠️ Success message not detected — verify manually in LinkedIn")

        await page.wait_for_timeout(5000)
        await page.screenshot(path=ss("post_success"))
        return True

    except Exception as e:
        logger.error(f"❌ Error during LinkedIn posting: {e}")
        await page.screenshot(path=ss("posting_error"))
        return False


async def upload_linkedin_image(page, image_path: str, ss):
    uploaded = False
    logger.info(f"📸 Uploading LinkedIn image: {image_path}")
    await page.screenshot(path=ss("before_image_upload"))

    # Try to open the media/photo uploader first.
    for sel in [
        "button[aria-label*='Add media']",
        "button[aria-label*='Add a photo']",
        "button[aria-label*='Photo']",
        "button:has-text('Add media')",
        "button:has-text('Media')",
        "button:has-text('Photo')"
    ]:
        try:
            btn = await page.wait_for_selector(sel, timeout=4000, state="visible")
            if btn:
                await btn.click(force=True)
                logger.info(f"✅ Media button clicked via: {sel}")
                await page.wait_for_timeout(1500)
                break
        except Exception:
            continue

    # Set file input. LinkedIn may keep this input hidden, so state="attached" is intentional.
    for retry in range(3):
        try:
            for sel in [
                "input[type='file'][accept*='image']",
                "input[type='file'][accept*='png']",
                "input[type='file'][accept*='jpg']",
                "input[type='file']"
            ]:
                try:
                    fi = await page.wait_for_selector(sel, state="attached", timeout=10000)
                    await fi.set_input_files(os.path.abspath(image_path))
                    logger.info(f"✅ Image file attached (retry {retry + 1})")
                    uploaded = True
                    break
                except PlaywrightTimeoutError:
                    continue
            if uploaded:
                break
            await page.wait_for_timeout(2000)
        except Exception as re_e:
            logger.debug(f"Upload retry {retry + 1}: {re_e}")

    if not uploaded:
        logger.error("❌ LinkedIn image upload failed")
        await page.screenshot(path=ss("image_upload_failed"))
        return False

    # Some LinkedIn upload flows show a Done/Next button in an editor dialog before returning to the composer.
    await page.wait_for_timeout(3000)
    
    next_btn_clicked = False
    for sel in [
        "button:has-text('Next')", 
        "button:has-text('Done')", 
        "button.share-box-footer__primary-btn",
        "span:text-is('Next')"
    ]:
        try:
            elements = page.locator("div[role='dialog']").locator(sel)
            for i in range(await elements.count()):
                btn = elements.nth(i)
                if await btn.is_visible():
                    # Wait for it to be enabled
                    for _ in range(10):
                        if (await btn.get_attribute("disabled")) is not None or \
                           (await btn.get_attribute("aria-disabled") == "true"):
                            await page.wait_for_timeout(1000)
                        else:
                            break
                            
                    # Let Playwright ensure actionability (no force=True)
                    await btn.click() 
                    logger.info(f"✅ Clicked '{sel}' to confirm image")
                    next_btn_clicked = True
                    break
        except Exception:
            pass
        if next_btn_clicked:
            break
            
    if not next_btn_clicked:
        logger.warning("⚠️ Did not find Next/Done button. Attempting fallback...")
        await click_by_text_or_role(page, ["Done", "Next"], timeout_ms=5000, required=False)

    # Wait for the editor dialog to transition back to the main composer
    await page.wait_for_timeout(2000)

    try:
        # Removed 'img' to prevent false positive matching of the editor dialog itself
        await page.wait_for_selector(
            "div.share-creation-state__preview, div[aria-label*='media preview']",
            timeout=15000
        )
        logger.info("✅ Image preview/upload confirmed")
    except PlaywrightTimeoutError:
        logger.warning("⚠️ Image preview not detected — continuing anyway")

    await page.wait_for_timeout(2000)
    await page.screenshot(path=ss("after_image_upload"))
    return True


async def click_by_text_or_role(page, names, timeout_ms=8000, required=True):
    """Small helper kept separate so post_to_linkedin stays close to your original structure."""
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

    custom_prompt = input("Custom prompt (press Enter to skip): ").strip() or "🚀 Automated LinkedIn Post!"

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
            user_data_dir="./linkedin_session",
            headless=headless,
            args=["--start-maximized"] if not headless else []
        )
        page = await browser.new_page()

        try:
            logger.info("🌐 Navigating to LinkedIn...")
            await page.goto("https://www.linkedin.com/feed/", wait_until="domcontentloaded", timeout=60000)
            await page.wait_for_timeout(3000)

            # Login check
            is_login = (
                "login" in page.url.lower() or
                "checkpoint" in page.url.lower() or
                "challenge" in page.url.lower() or
                "signup" in page.url.lower() or
                await page.query_selector("input[name='session_key'], input#username") is not None
            )
            if is_login:
                await wait_for_login(page)
            else:
                logger.info("✅ Already logged in via saved LinkedIn session.")

            # Post loop
            for day in range(num_days):
                logger.info(f"📅 Day {day + 1}/{num_days}")
                post_text  = generate_post_text(client, domain_name, custom_prompt, day + 1)
                image_path = generate_image_with_huggingface(image_prompt) if want_image else None
                success    = await post_to_linkedin(page, post_text, image_path)

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
