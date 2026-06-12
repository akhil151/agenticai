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

# â”€â”€ Directories â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
IMAGE_CACHE_DIR = "./image_cache"
STATIC_DIR      = "./static"
SCREENSHOTS_DIR = "./screenshots/facebook"
LOGS_DIR        = "./logs"

for _d in [IMAGE_CACHE_DIR, STATIC_DIR, SCREENSHOTS_DIR, LOGS_DIR, "./fb_session"]:
    os.makedirs(_d, exist_ok=True)

# â”€â”€ Logging â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"{LOGS_DIR}/facebook_automation.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# â”€â”€ Config â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
GROQ_MODEL    = "llama-3.3-70b-versatile"


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Image helpers
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def generate_image_with_huggingface(prompt: str) -> str:
    """Generate image via local Stable Diffusion to ensure perfect execution on 4GB VRAM."""
    from local_image_generator import generate_local_image
    # We pass create_placeholder_image as the fallback so it never fails
    path = generate_local_image(prompt, IMAGE_CACHE_DIR, create_placeholder_image)
    if path:
        cleanup_cache()
    return path


def create_placeholder_image() -> str:
    """Simple grey placeholder so upload never fails with a missing file."""
    path = os.path.join(STATIC_DIR, "placeholder.png")
    try:
        img = Image.new("RGB", (512, 512), color="lightgray")
        img.save(path, format="PNG")
        logger.info(f"Created placeholder at {path}")
    except Exception as e:
        logger.error(f"Placeholder creation failed: {e}")
    return path


def cleanup_cache(max_files: int = 100):
    files = sorted(glob.glob(os.path.join(IMAGE_CACHE_DIR, "*.png")), key=os.path.getctime)
    for old in files[:-max_files]:
        os.remove(old)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Text helper
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def generate_post_text(client: Groq, domain_name: str, custom_prompt: str, day: int) -> str:
    prompt = f"""
Generate a short, engaging Facebook post about "{custom_prompt}" in the context of "{domain_name}".
Make it exciting, add relevant emojis, and keep it under 200 words.
Include "#Day{day}" at the end to make it part of a series.
"""
    try:
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates engaging social media posts."},
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
        return f"ðŸš€ Default post about {custom_prompt} (via {domain_name}) #Day{day}"


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Login wait
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

async def wait_for_login(page):
    """Poll every 3 s until Facebook home feed is accessible."""
    logger.info("ðŸ”‘ Facebook login required.")
    logger.info("ðŸ‘‰ Please log in manually in the browser â€” the script continues automatically once done...")
    while True:
        try:
            url = page.url
            on_login = (
                "login"      in url.lower() or
                "checkpoint" in url.lower() or
                await page.query_selector("input[name='email']") is not None
            )
            if not on_login:
                try:
                    await page.wait_for_selector(
                        "div[role='feed'], div[aria-label='News Feed'], div[data-pagelet='FeedUnit_0']",
                        timeout=5000
                    )
                    logger.info("âœ… Login confirmed â€” continuing...")
                    return
                except PlaywrightTimeoutError:
                    pass
        except Exception:
            pass
        await asyncio.sleep(3)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Posting
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

async def post_to_facebook(page, post_text: str, image_path: str = None):
    ss = lambda name: f"{SCREENSHOTS_DIR}/{name}.png"
    try:
        # â”€â”€ Page state snapshot â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        await page.screenshot(path=ss("00_page_before_compose"))
        logger.info(f"ðŸ“Œ Current URL: {page.url}")

        # â”€â”€ Dismiss any leftover 'Save as draft?' dialogs â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        try:
            draft_btn = page.locator("div[role='button']:has-text('Delete draft'), span:has-text('Delete draft')")
            if await draft_btn.count() > 0 and await draft_btn.first.is_visible():
                await draft_btn.first.click(force=True)
                logger.info("ðŸ—‘ï¸ Dismissed 'Save as draft' dialog")
                await page.wait_for_timeout(1500)
        except Exception as e:
            logger.debug(f"Draft dialog check: {e}")

        # â”€â”€ Open composer â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        logger.info("ðŸ“ Opening post composer...")
        composer_opened = False

        # Strategy 1: click "What's on your mind" button
        for sel in [
            "div[role='button'][aria-label*=\"What's on your mind\"]",
            "div[aria-label*=\"What's on your mind\"]",
            "div[role='button']:has-text(\"What's on your mind\")",
            "span:has-text(\"What's on your mind\")",
        ]:
            try:
                el = await page.wait_for_selector(sel, timeout=5000, state="visible")
                if el:
                    await el.click(force=True)
                    logger.info(f"âœ… Composer opened via: {sel}")
                    composer_opened = True
                    break
            except PlaywrightTimeoutError:
                logger.debug(f"   Not found: {sel}")
            except Exception as e:
                logger.debug(f"   Error '{sel}': {e}")

        # Strategy 2: textbox already visible (inline composer state)
        if not composer_opened:
            try:
                tb = page.locator("div[role='textbox']")
                if await tb.count() > 0 and await tb.first.is_visible():
                    logger.info("âœ… Textbox already visible â€” composer already open")
                    composer_opened = True
            except Exception as e:
                logger.debug(f"Textbox pre-check: {e}")

        # Strategy 3: reload page and retry once
        if not composer_opened:
            logger.warning("âš ï¸ Composer not found â€” reloading and retrying...")
            await page.goto("https://www.facebook.com", wait_until="domcontentloaded")
            await page.wait_for_timeout(5000)
            await page.screenshot(path=ss("01_page_after_reload"))
            for sel in [
                "div[role='button'][aria-label*=\"What's on your mind\"]",
                "div[role='button']:has-text(\"What's on your mind\")",
                "span:has-text(\"What's on your mind\")",
            ]:
                try:
                    el = await page.wait_for_selector(sel, timeout=8000, state="visible")
                    if el:
                        await el.click(force=True)
                        logger.info(f"âœ… Composer opened after reload via: {sel}")
                        composer_opened = True
                        break
                except PlaywrightTimeoutError:
                    logger.debug(f"   Post-reload not found: {sel}")
                except Exception as e:
                    logger.debug(f"   Post-reload error '{sel}': {e}")

        if not composer_opened:
            try:
                btns = await page.evaluate("""() =>
                    Array.from(document.querySelectorAll('[role="button"]'))
                        .slice(0, 20)
                        .map(b => ({text: b.textContent.trim().substring(0,40), label: b.getAttribute('aria-label')}))
                """)
                logger.error(f"[DIAGNOSTIC] Visible buttons: {btns}")
            except Exception as de:
                logger.error(f"[DIAGNOSTIC] Could not list buttons: {de}")
            logger.error("âŒ Could not open composer after all strategies")
            await page.screenshot(path=ss("editor_button_not_found"))
            return False

        await page.wait_for_timeout(2000)

        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        # STEP 1 â€” Upload image FIRST (before typing caption)
        # ROOT CAUSE: Facebook wipes the textbox when you click 'Photo/video'
        # because it transitions to a new UI state. Uploading first and typing
        # the caption last guarantees the caption is never lost.
        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        uploaded = False
        if image_path and os.path.exists(image_path):
            logger.info(f"ðŸ“¸ Uploading image: {image_path}")
            await page.screenshot(path=ss("before_image_upload"))

            try:
                await page.wait_for_selector("div[aria-label='Photo/video']", timeout=10000, state="visible")
                logger.info("âœ… Toolbar loaded (Photo/video visible)")
            except PlaywrightTimeoutError:
                logger.warning("âš ï¸ Toolbar not detected; waiting 3s")
                await page.wait_for_timeout(3000)

            for retry in range(3):
                if uploaded:
                    break
                try:
                    # Priority 1: inject directly into existing file input
                    for sel in ["input[type='file'][accept*='image']",
                                "input.x1s85apg[type='file']",
                                "input[type='file'][multiple]"]:
                        try:
                            fi = await page.wait_for_selector(sel, timeout=3000)
                            await fi.set_input_files(os.path.abspath(image_path))
                            logger.info(f"âœ… Image via direct input (retry {retry+1})")
                            uploaded = True
                            break
                        except PlaywrightTimeoutError:
                            continue
                    if uploaded:
                        break

                    # Priority 2: click Photo/video â†’ wait for file input â†’ inject
                    photo_clicked = False
                    for sel in ["div[aria-label='Photo/video']",
                                "div[role='button'][aria-label*='Photo']"]:
                        try:
                            trigger = await page.wait_for_selector(sel, timeout=5000, state="visible")
                            await trigger.scroll_into_view_if_needed()
                            await trigger.click(force=True)
                            logger.info(f"âœ… Clicked Photo/video via {sel}")
                            photo_clicked = True
                            break
                        except PlaywrightTimeoutError:
                            logger.debug(f"   Photo/video not found: {sel}")
                        except Exception as e:
                            logger.debug(f"   Photo/video error '{sel}': {e}")

                    if photo_clicked:
                        await page.wait_for_timeout(1500)
                        for fi_sel in ["input[type='file'][accept*='image']",
                                       "input[type='file'][multiple]",
                                       "input[type='file']"]:
                            try:
                                fi = page.locator(fi_sel).first
                                if await fi.count() > 0:
                                    await fi.set_input_files(os.path.abspath(image_path))
                                    logger.info(f"âœ… Image injected via {fi_sel} (retry {retry+1})")
                                    uploaded = True
                                    break
                            except Exception as fie:
                                logger.debug(f"   Inject error '{fi_sel}': {fie}")

                    # Priority 3: file chooser fallback
                    if not uploaded:
                        for sel in ["div[aria-label='Photo/video']",
                                    "div[role='button'][aria-label*='Photo']"]:
                            try:
                                trigger = await page.wait_for_selector(sel, timeout=5000, state="visible")
                                async with page.expect_file_chooser(timeout=8000) as fc_info:
                                    await trigger.click(force=True)
                                fc = await fc_info.value
                                await fc.set_files(os.path.abspath(image_path))
                                logger.info(f"âœ… Image via file chooser (retry {retry+1})")
                                uploaded = True
                                break
                            except PlaywrightTimeoutError:
                                logger.debug(f"   File chooser timeout: {sel}")
                            except Exception as fce:
                                logger.debug(f"   File chooser error '{sel}': {fce}")

                    if uploaded:
                        break
                    await page.wait_for_timeout(2000)

                except Exception as re_e:
                    logger.warning(f"âš ï¸ Upload retry {retry+1} exception: {re_e}")
                    await page.screenshot(path=ss(f"upload_retry_{retry+1}_error"))

            if uploaded:
                try:
                    await page.wait_for_selector(
                        "div[role='dialog'] img[src*='blob:'], "
                        "div[role='dialog'] img[src*='scontent'], "
                        "div[role='dialog'] img[src*='data:'], "
                        "div[role='dialog'] img[class*='img']",
                        timeout=15000
                    )
                    logger.info("âœ… Image preview confirmed in dialog")
                except PlaywrightTimeoutError:
                    logger.warning("âš ï¸ Preview selector didn't match â€” image was injected, continuing")
                await page.wait_for_timeout(3000)
                await page.screenshot(path=ss("after_image_upload"))
            else:
                logger.warning("âš ï¸ All upload retries failed; posting without image")
        else:
            logger.info("No image provided; skipping upload")

        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        # STEP 2 â€” Type caption AFTER image upload settles
        # CRITICAL: Must be done LAST so Facebook's UI transition doesn't wipe it
        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        logger.info("âŒ¨ Filling caption (after image upload)...")
        textbox = None
        for tb_sel in ["div[role='textbox']", "[contenteditable='true']", "div[data-lexical-editor='true']"]:
            try:
                textbox = await page.wait_for_selector(tb_sel, timeout=10000, state="visible")
                if textbox:
                    logger.info(f"âœ… Textbox found via: {tb_sel}")
                    break
            except PlaywrightTimeoutError:
                logger.debug(f"   Textbox not found: {tb_sel}")
            except Exception as e:
                logger.debug(f"   Textbox error '{tb_sel}': {e}")

        if not textbox:
            logger.error("âŒ Could not locate text editor after image upload")
            await page.screenshot(path=ss("text_fill_failed"))
            return False

        await textbox.click()
        await page.wait_for_timeout(500)

        try:
            await textbox.fill(post_text)
            logger.info("âœ… Caption filled via fill()")
        except Exception as fill_err:
            logger.warning(f"âš ï¸ fill() failed ({fill_err}) â€” trying type() fallback")
            try:
                await textbox.type(post_text, delay=10)
                logger.info("âœ… Caption typed via type() fallback")
            except Exception as e2:
                logger.error(f"âŒ Could not fill caption: {e2}")
                await page.screenshot(path=ss("text_fill_failed"))
                return False

        await page.wait_for_timeout(1000)
        await page.screenshot(path=ss("after_text_fill"))

        # â”€â”€ Click Post button â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # Diagnostic confirmed: DIV with role='button' and aria-label='Post'
        logger.info("ðŸš€ Clicking Post button...")
        await page.screenshot(path=ss("before_post_click"))
        post_success = False

        start_time = datetime.now().timestamp()
        while datetime.now().timestamp() - start_time < 30 and not post_success:
            try:
                post_locator = page.locator("div[aria-label='Post'][role='button']")
                count = await post_locator.count()
                for i in range(count - 1, -1, -1):
                    btn = post_locator.nth(i)
                    try:
                        if not await btn.is_visible():
                            continue
                        aria_disabled = await btn.get_attribute("aria-disabled")
                        if aria_disabled and aria_disabled.lower() == "true":
                            continue
                        box = await btn.bounding_box()
                        if not box:
                            continue
                        cx = box["x"] + box["width"] / 2
                        cy = box["y"] + box["height"] / 2
                        await page.mouse.move(cx, cy)
                        await page.wait_for_timeout(200)
                        await page.mouse.click(cx, cy)
                        logger.info(f"âœ… Post button clicked at ({cx:.0f}, {cy:.0f})")
                        post_success = True
                        break
                    except Exception:
                        continue
            except Exception as e:
                logger.debug(f"Post button scan error: {e}")
            if not post_success:
                await page.wait_for_timeout(2000)

        if not post_success:
            logger.error("âŒ Failed to click Post button")
            await page.screenshot(path=ss("post_click_failed"))
            return False

        await page.wait_for_timeout(5000)
        logger.info("âœ… Post published successfully!")
        await page.screenshot(path=ss("post_success"))
        return True

    except Exception as e:
        logger.error(f"âŒ Error during posting: {e}")
        await page.screenshot(path=ss("posting_error"))
        return False


        await page.wait_for_timeout(5000)
        logger.info("âœ… Post published successfully!")
        await page.screenshot(path=ss("post_success"))
        return True

    except Exception as e:
        logger.error(f"âŒ Error during posting: {e}")
        await page.screenshot(path=ss("posting_error"))
        return False


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Entry point
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

async def run():
    # â”€â”€ Collect inputs â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    domain_name = input("Domain Name: ").strip()
    if not domain_name:
        logger.error("Domain Name is required.")
        return

    custom_prompt = input("Custom prompt (press Enter to skip): ").strip() or "ðŸš€ Automated Facebook Post!"

    want_image = input("Do you want an image? (y/n): ").strip().lower() == "y"
    image_prompt = ""
    if want_image:
        image_prompt = input("Image prompt: ").strip()
        if not image_prompt:
            logger.error("Image prompt is required when image is enabled.")
            return

    num_days_raw = input("Number of days (default 1): ").strip()
    num_days = int(num_days_raw) if num_days_raw.isdigit() else 1

    # â”€â”€ Groq client â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        logger.error("âŒ GROQ_API_KEY not found in .env â€” please add it and retry.")
        return
    logger.info("âœ… Groq API key loaded from .env")
    client = Groq(api_key=groq_api_key)

    # â”€â”€ Browser â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    async with async_playwright() as p:
        headless = os.getenv("HEADLESS", "false").lower() == "true"
        browser  = await p.chromium.launch_persistent_context(
            user_data_dir="./fb_session",
            headless=headless,
            args=["--start-maximized"] if not headless else []
        )
        page = await browser.new_page()

        try:
            logger.info("ðŸŒ Navigating to Facebook...")
            await page.goto("https://www.facebook.com", wait_until="domcontentloaded", timeout=60000)
            await page.wait_for_timeout(3000)

            # Login check
            is_login = (
                "login" in page.url.lower() or
                await page.query_selector("input[name='email']") is not None
            )
            if is_login:
                await wait_for_login(page)
            else:
                logger.info("âœ… Already logged in via saved session.")

            # Post loop
            for day in range(num_days):
                logger.info(f"ðŸ“… Day {day + 1}/{num_days}")
                post_text  = generate_post_text(client, domain_name, custom_prompt, day + 1)
                image_path = generate_image_with_huggingface(image_prompt) if want_image else None
                success    = await post_to_facebook(page, post_text, image_path)

                if success:
                    logger.info(f"âœ… Day {day + 1} posted successfully")
                else:
                    logger.error(f"âŒ Day {day + 1} post failed")

                if day < num_days - 1:
                    logger.info("â³ Pausing before next post...")
                    await page.wait_for_timeout(5000)

        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            await page.screenshot(path=f"{SCREENSHOTS_DIR}/unexpected_error.png")
        finally:
            await page.wait_for_timeout(5000)
            await browser.close()


if __name__ == "__main__":
    asyncio.run(run())
