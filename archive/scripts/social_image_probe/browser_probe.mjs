// Anonymous browser diagnostics using Node 22's built-in WebSocket and local Chromium.
// No model API, browser profile reuse, credentials, stealth or scrolling.
import fs from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import {spawn} from 'node:child_process';

const [browser, output, ...instagramTags] = process.argv.slice(2);
if (!browser || !output) throw new Error('Usage: node browser_probe.mjs CHROMIUM NEW_OUTPUT_DIR');
process.umask(0o077);
await fs.mkdir(output, {recursive: false, mode: 0o700});
const profile = await fs.mkdtemp(path.join(os.tmpdir(), 'social-cdp-probe-'));
const args = ['--headless', '--no-sandbox', '--disable-dev-shm-usage', '--no-first-run',
  '--disable-background-networking', '--disable-component-update', '--disable-sync',
  '--remote-debugging-address=127.0.0.1', '--remote-debugging-port=0',
  `--user-data-dir=${profile}`];
const proxy = process.env.HTTPS_PROXY || process.env.https_proxy;
if (proxy) {
  if (new URL(proxy).username) throw new Error('Authenticated proxy is not supported');
  args.push(`--proxy-server=${proxy}`);
}
args.push('about:blank');
const child = spawn(browser, args, {detached: true, stdio: ['ignore', 'ignore', 'pipe']});
let stderr = '', ws, requestId = 0;
const pending = new Map();
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));
const started = Date.now();
child.stderr.on('data', data => { stderr += data.toString(); });
function call(method, params = {}, sessionId) {
  return new Promise((resolve, reject) => {
    const id = ++requestId;
    const timer = setTimeout(() => { pending.delete(id); reject(new Error(`CDP timeout: ${method}`)); }, 20000);
    pending.set(id, {resolve, reject, timer});
    ws.send(JSON.stringify({id, method, params, ...(sessionId ? {sessionId} : {})}));
  });
}
try {
  while (!stderr.includes('DevTools listening on ') && Date.now() - started < 15000) {
    if (child.exitCode !== null) throw new Error(`Chromium exited: ${child.exitCode}`);
    await pause(100);
  }
  const endpoint = stderr.match(/DevTools listening on (ws:\/\/\S+)/)?.[1];
  if (!endpoint) throw new Error('Chromium did not expose CDP');
  ws = new WebSocket(endpoint);
  await new Promise((resolve, reject) => { ws.onopen = resolve; ws.onerror = reject; });
  ws.onmessage = event => {
    const data = JSON.parse(event.data), entry = pending.get(data.id);
    if (!entry) return;
    pending.delete(data.id); clearTimeout(entry.timer);
    if (data.error) entry.reject(new Error(JSON.stringify(data.error)));
    else entry.resolve(data.result);
  };
  let targets;
  if (instagramTags.length === 1 && instagramTags[0].startsWith('--targets=')) {
    const entries = JSON.parse(await fs.readFile(instagramTags[0].slice('--targets='.length), 'utf8'));
    if (!Array.isArray(entries) || entries.length > 12) throw new Error('Expected at most 12 targets');
    targets = entries.map(({tag, url}) => {
      if (!/^[\w-]{1,64}$/.test(tag) || new URL(url).origin !== 'https://www.instagram.com') {
        throw new Error('Targets must have a safe label and public Instagram URL');
      }
      return ['instagram', tag, url];
    });
  } else {
  if (instagramTags.some(tag => !/^[\w-]{1,64}$/.test(tag))) throw new Error('Invalid tag');
  targets = instagramTags.length ? instagramTags.map(tag =>
    ['instagram', tag, `https://www.instagram.com/explore/tags/${encodeURIComponent(tag)}/`]) : [
    ['instagram', 'portrait', 'https://www.instagram.com/explore/tags/portrait/'],
    ['xiaohongshu', 'landscape', 'https://www.xiaohongshu.com/search_result?keyword=%E9%A3%8E%E6%99%AF&source=web_search_result_notes'],
    ['xiaohongshu', 'portrait', 'https://www.xiaohongshu.com/search_result?keyword=%E5%86%99%E7%9C%9F&source=web_search_result_notes'],
  ];
  }
  const results = [];
  for (const [source, tag, url] of targets) {
    const {targetId} = await call('Target.createTarget', {url: 'about:blank'});
    const {sessionId} = await call('Target.attachToTarget', {targetId, flatten: true});
    let result = {source, tag, requested_url: url};
    try {
      await call('Page.enable', {}, sessionId);
      result.navigation = await call('Page.navigate', {url}, sessionId);
      await pause(12000);
      const evaluated = await call('Runtime.evaluate', {returnByValue: true, expression: `JSON.stringify({
        final_url: location.href, title: document.title,
        body_text: document.body?.innerText.slice(0, 5000) || '',
        images: [...document.images].map(image => ({
          src: image.currentSrc || image.src, alt: image.alt,
          srcset: image.srcset,
          href: image.closest('a')?.href || '',
          width: image.naturalWidth, height: image.naturalHeight
        })),
        video_count: document.querySelectorAll('video').length,
        post_links: [...document.querySelectorAll('a[href]')].map(a => a.href)
          .filter(h => /\\/(p|reel|explore)\\/[^/?]+/.test(h))
      })`}, sessionId);
      if (evaluated.exceptionDetails) throw new Error('Page evaluation failed');
      Object.assign(result, JSON.parse(evaluated.result.value));
      const dom = await call('Runtime.evaluate', {expression: 'document.documentElement.outerHTML', returnByValue: true}, sessionId);
      await fs.writeFile(path.join(output, `${source}_${tag}.html`), dom.result.value);
    } catch (error) { result.error = String(error); }
    await call('Target.closeTarget', {targetId});
    results.push(result);
    await fs.writeFile(path.join(output, `${source}_${tag}.json`), JSON.stringify(result, null, 2));
    console.log(JSON.stringify({source, tag, final_url: result.final_url, title: result.title,
      image_count: result.images?.length, post_link_count: result.post_links?.length, error: result.error}));
  }
  await fs.writeFile(path.join(output, 'summary.json'), JSON.stringify({seconds: (Date.now()-started)/1000,
    model_api_calls: 0, authentication: 'none', scrolls: 0, wait_ms_per_page: 12000, results}, null, 2));
} finally {
  ws?.close();
  for (const {timer} of pending.values()) clearTimeout(timer);
  try { process.kill(-child.pid, 'SIGTERM'); } catch {}
  await pause(1000);
  try { process.kill(-child.pid, 'SIGKILL'); } catch {}
  await fs.writeFile(path.join(output, 'chromium.stderr.log'), stderr);
  await fs.rm(profile, {recursive: true, force: true});
}
