"""Isolated browser checks. Never reads or saves real human decisions.

Run with a Python environment containing Playwright and MANUAL_REVIEW_CHROME
pointing to a Chromium executable if the Playwright default is unavailable.
Every fixture and browser context is temporary and explicitly test_only.
"""
from __future__ import annotations

import base64
import copy
import json
import os
from pathlib import Path
import re
import struct
import tempfile
import unittest
import zlib

try:
    from playwright.sync_api import expect, sync_playwright
except ImportError:
    sync_playwright = None


def tiny_png() -> str:
    def chunk(kind, value):
        return struct.pack('!I', len(value)) + kind + value + struct.pack('!I', zlib.crc32(kind + value))
    value = b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('!2I5B', 2, 2, 8, 2, 0, 0, 0))
    value += chunk(b'IDAT', zlib.compress((b'\0' + b'\xff\0\0' * 2) * 2)) + chunk(b'IEND', b'')
    return 'data:image/png;base64,' + base64.b64encode(value).decode()


def fixture():
    malicious = '</script><script>window.TEST_ONLY_XSS_EXECUTED=true</script><img src=x onerror="window.TEST_ONLY_XSS_EXECUTED=true">'
    return {'queue_id': 'test_only_manual_review_browser_queue', 'title': 'TEST ONLY', 'items': [
        {'item_id': f'test_only_{i}', 'review_number': i, 'image_data_url': tiny_png(),
         'source_name': 'test_only_source_A' if i % 2 else 'test_only_source_B',
         'image_kind': 'photograph' if i % 2 else 'chart', 'source_path': malicious,
         'source_member': f'test_only_{i}.png', 'source_url': 'javascript:window.TEST_ONLY_XSS_EXECUTED=true',
         'image_path': f'images/test_only_{i}.png', 'reason': malicious,
         'alternative_explanation': malicious, 'width': 2, 'height': 2,
         'preview_width': 2, 'preview_height': 2,
         'visible_evidence': [{'location': malicious, 'observation': malicious, 'supports': 'ambiguous'}]}
        for i in range(1, 5)]}


@unittest.skipIf(sync_playwright is None, 'Playwright is an optional browser-test dependency')
class ManualReviewBrowserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='test_only_manual_review_')
        cls.root = Path(cls.temp.name)
        template = Path(__file__).with_name('manual_review_template.html').read_text()
        cls.data = fixture()
        encoded = json.dumps(cls.data, ensure_ascii=False).replace('<', '\\u003c')
        cls.html = cls.root / 'test_only_index.html'
        cls.html.write_text(template.replace('__REVIEW_DATA_JSON__', encoded))
        cls.pw = sync_playwright().start()
        chrome = os.environ.get('MANUAL_REVIEW_CHROME', '/root/.cache/ms-playwright/chromium-1248/chrome-linux64/chrome')
        cls.browser = cls.pw.chromium.launch(executable_path=chrome, headless=True, args=['--no-sandbox'])

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.pw.stop()
        cls.temp.cleanup()

    def setUp(self):
        self.context = self.browser.new_context(viewport={'width': 1450, 'height': 950}, accept_downloads=True)
        self.page = self.context.new_page()
        self.errors = []
        self.page.on('pageerror', lambda err: self.errors.append(str(err)))
        self.page.goto(self.html.as_uri())
        self.page.locator('#position').wait_for()

    def tearDown(self):
        self.assertEqual([], self.errors)
        self.context.close()

    def export(self):
        with self.page.expect_download() as event:
            self.page.locator('#exportButton').click()
        return json.loads(Path(event.value.path()).read_text())

    def import_payload(self, payload):
        self.page.locator('#importFile').set_input_files({
            'name': 'test_only_decisions.json', 'mimeType': 'application/json',
            'buffer': json.dumps(payload).encode()})
        expect(self.page.locator('#operationMessage')).to_have_text(re.compile(r'^(导入失败|已导入|已取消导入)'))

    def test_defaults_empty_and_images_decode(self):
        self.assertEqual('0 / 4', self.page.locator('#processedCount').inner_text())
        self.assertEqual(0, self.page.locator('[data-decision][aria-pressed="true"]').count())
        self.assertEqual([], self.export()['decisions'])
        self.assertEqual(2, self.page.locator('#mainImage').evaluate('(el) => el.naturalWidth'))

    def test_save_refresh_restore_and_note_draft(self):
        self.page.locator('#note').fill('test_only retained note')
        self.page.locator('[data-decision="keep"]').click()
        self.page.locator('#nextButton').click()
        self.page.locator('#note').fill('test_only undecided draft')
        self.page.reload()
        self.assertEqual('2 / 4', self.page.locator('#position').inner_text())
        self.assertEqual('test_only undecided draft', self.page.locator('#note').input_value())
        self.assertEqual('1 / 4', self.page.locator('#processedCount').inner_text())
        self.page.locator('#prevButton').click()
        self.assertEqual('true', self.page.locator('[data-decision="keep"]').get_attribute('aria-pressed'))
        result = self.export()
        self.assertEqual(1, len(result['decisions']))
        self.assertEqual('test_only retained note', result['decisions'][0]['note'])

    def test_keyboard_input_focus_and_all_decisions(self):
        self.page.locator('#note').focus()
        self.page.keyboard.type('1234')
        self.page.keyboard.press('ArrowRight')
        self.assertEqual('1 / 4', self.page.locator('#position').inner_text())
        self.assertEqual('0 / 4', self.page.locator('#processedCount').inner_text())
        self.page.locator('#title').click()
        for key, decision in [('1', 'keep'), ('2', 'exclude_ai'), ('3', 'exclude_quality'), ('4', 'unsure')]:
            self.page.keyboard.press(key)
            self.assertEqual('true', self.page.locator(f'[data-decision="{decision}"]').get_attribute('aria-pressed'))
        self.page.keyboard.press('ArrowRight')
        self.assertEqual('2 / 4', self.page.locator('#position').inner_text())

    def test_filters_preserve_fixed_numbers(self):
        self.page.locator('#kindFilter').select_option('photograph')
        self.assertEqual(['#1', '#3'], self.page.locator('#directoryList strong').all_inner_texts())
        self.page.locator('#nextButton').click()
        self.assertEqual('3 / 4', self.page.locator('#position').inner_text())
        self.page.locator('[data-decision="exclude_quality"]').click()
        self.page.locator('#statusFilter').select_option('unreviewed')
        self.assertEqual(['#1'], self.page.locator('#directoryList strong').all_inner_texts())
        self.page.locator('#jumpNumber').fill('4')
        self.page.locator('#jumpForm button').click()
        self.assertEqual('4 / 4', self.page.locator('#position').inner_text())
        self.assertEqual('all', self.page.locator('#statusFilter').input_value())

    def test_import_export_roundtrip_and_invalid_files_atomic(self):
        self.page.locator('[data-decision="keep"]').click()
        base = self.export()
        self.page.locator('#clearDecision').click()
        self.import_payload(base)
        self.assertEqual('1 / 4', self.page.locator('#processedCount').inner_text())
        invalid = []
        changed = copy.deepcopy(base); changed['queue_id'] = 'test_only_wrong_queue'; invalid.append(changed)
        changed = copy.deepcopy(base); changed['decisions'][0]['item_id'] = 'test_only_unknown'; invalid.append(changed)
        changed = copy.deepcopy(base); changed['decisions'].append(changed['decisions'][0].copy()); invalid.append(changed)
        changed = copy.deepcopy(base); changed['decisions'][0]['review_number'] = 2; invalid.append(changed)
        changed = copy.deepcopy(base); changed['decisions'][0]['decision'] = 'auto_admit'; invalid.append(changed)
        for value in invalid:
            self.page.locator('#operationMessage').evaluate('(el) => el.textContent = ""')
            self.import_payload(value)
            self.assertIn('导入失败', self.page.locator('#operationMessage').inner_text())
            self.assertEqual(base['decisions'], self.export()['decisions'])

    def test_source_text_does_not_execute_and_unsafe_link_hidden(self):
        self.assertIsNone(self.page.evaluate('window.TEST_ONLY_XSS_EXECUTED'))
        self.assertIn('<script>', self.page.locator('#reason').text_content())
        self.assertTrue(self.page.locator('#sourceLink').is_hidden())
        self.assertIsNone(self.page.locator('#sourceLink').get_attribute('href'))
        self.assertEqual(0, self.page.locator('#sourcePath img, #reason script').count())

    def test_conflicting_import_requires_confirmation(self):
        self.page.locator('#note').fill('test_only existing judgment')
        self.page.locator('[data-decision="keep"]').click()
        existing = self.export()
        incoming = copy.deepcopy(existing)
        incoming['decisions'][0]['decision'] = 'exclude_ai'
        incoming['decisions'][0]['note'] = 'test_only conflicting judgment'
        self.page.once('dialog', lambda dialog: dialog.dismiss())
        self.import_payload(incoming)
        self.assertEqual(existing['decisions'], self.export()['decisions'])
        self.page.once('dialog', lambda dialog: dialog.accept())
        self.import_payload(incoming)
        self.assertEqual(incoming['decisions'], self.export()['decisions'])

    def test_unreviewed_auto_next_does_not_skip(self):
        self.page.locator('#statusFilter').select_option('unreviewed')
        self.page.locator('#autoNext').check()
        for number in range(1, 5):
            self.assertEqual(f'{number} / 4', self.page.locator('#position').inner_text())
            self.page.locator('[data-decision="unsure"]').click()
        self.assertEqual('4 / 4', self.page.locator('#processedCount').inner_text())
        self.assertTrue(self.page.locator('#emptyState').is_visible())
        self.assertEqual([1, 2, 3, 4], [r['review_number'] for r in self.export()['decisions']])

    def test_zoom_disables_decision_hotkeys(self):
        self.page.locator('#mainImage').click()
        self.assertTrue(self.page.locator('#zoomDialog').is_visible())
        self.page.keyboard.press('2')
        self.assertEqual('0 / 4', self.page.locator('#processedCount').inner_text())
        self.page.locator('#closeZoom').click()

    def test_storage_failure_keeps_export_available(self):
        self.page.evaluate("() => { Storage.prototype.setItem = function() { throw new Error('test_only storage quota failure'); }; }")
        self.page.locator('[data-decision="keep"]').click()
        self.assertTrue(self.page.locator('#storageWarning').is_visible())
        self.assertIn('暂存失败', self.page.locator('#storageWarning').inner_text())
        self.assertEqual('keep', self.export()['decisions'][0]['decision'])


if __name__ == '__main__':
    unittest.main()
