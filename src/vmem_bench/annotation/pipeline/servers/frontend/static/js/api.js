(function () {
  'use strict';

  // ONLY real SSO/the reverse proxy login markers. A generic HTML page (e.g. a gateway
  // 502/504 timeout page shown when the backend is briefly slow under rerun load)
  // must NOT be mistaken for a logout — that produced a bogus "SSO expired →
  // redirect" every time /api/health got slow. Keep this strict.
  function looksLikeSsoLoginHtml(text) {
    if (typeof text !== 'string') return false;
    const head = text.slice(0, 2000).toLowerCase();
    return (
      head.includes('sso-p') ||
      head.includes('sso.corp.example.org') ||
      head.includes('sso') ||
      head.includes('cas/login')
    );
  }

  function looksLikeHtml(text) {
    if (typeof text !== 'string') return false;
    const head = text.slice(0, 200).toLowerCase();
    return head.includes('<!doctype html') || head.includes('<html');
  }

  function ssoError(status, loginUrl) {
    const err = new Error(
      'remote 网关登录已失效（the reverse proxy → SSO）。即将跳转internal SSO 登录页；登录后会回到本控制台。'
    );
    err.code = 'SSO_REQUIRED';
    err.status = status || 401;
    err.loginUrl = loginUrl || '';
    return err;
  }

  // Non-SSO, non-JSON response: usually the reverse proxy returning its own HTML
  // error page because the backend was briefly slow (KFS pressure after a
  // rerun). This is transient — surface it as "busy, retrying", never as SSO.
  function gatewayBusyError(status) {
    const err = new Error(
      '网关/后端暂时繁忙，返回了非 JSON 页面（常见于点「重跑」后负载升高、网关先超时）。' +
      '这不是登录失效——稍等几秒会自动重试；若持续，请硬刷新页面。'
    );
    err.code = 'GATEWAY_BUSY';
    err.status = status || 502;
    return err;
  }

  async function request(method, url, body) {
    const opts = {
      method: method,
      // Keep the reverse proxy session cookie on the remote HTTPS host.
      credentials: 'same-origin',
      // Do not follow 302 → sso.corp.example.org into an HTML login page.
      redirect: 'manual',
      headers: { Accept: 'application/json' }
    };
    if (body !== undefined) {
      opts.headers['Content-Type'] = 'application/json';
      opts.body = JSON.stringify(body);
    }
    const resp = await fetch(url, opts);

    if (resp.type === 'opaqueredirect') {
      throw ssoError(0, '');
    }
    if (resp.status >= 300 && resp.status < 400) {
      const loc = resp.headers.get('Location') || '';
      if (/sso\.corp\.internal\.com|sso|cas\/login/i.test(loc)) {
        throw ssoError(resp.status, loc);
      }
      throw new Error('unexpected redirect (' + resp.status + '): ' + loc.slice(0, 120));
    }

    const ctype = resp.headers.get('Content-Type') || '';
    const payload = ctype.includes('application/json')
      ? await resp.json()
      : await resp.text();

    // Distinguish a real SSO login page from a generic gateway error page.
    if (typeof payload === 'string' && looksLikeSsoLoginHtml(payload)) {
      throw ssoError(resp.status, '');
    }
    if (typeof payload === 'string' && looksLikeHtml(payload)) {
      throw gatewayBusyError(resp.status);
    }

    if (!resp.ok) {
      let message =
        (payload && payload.error) ||
        (typeof payload === 'string' ? payload : resp.statusText) ||
        'request failed';
      if (typeof message === 'string' && /504|Gateway Time-?out/i.test(message)) {
        message =
          '请求超时（504）。多为点「重跑」后负载升高、后端目录扫描变慢所致；' +
          '稍等几秒后会自动重试，通常无需重新登录。若持续，请硬刷新页面。';
      } else if (typeof message === 'string' && message.length > 180) {
        message = message.slice(0, 180) + '…';
      }
      const err = new Error(message);
      err.status = resp.status;
      err.payload = payload;
      throw err;
    }
    return payload;
  }

  function q(params) {
    return Object.entries(params)
      .filter(([, v]) => v !== undefined && v !== null && v !== '')
      .map(([k, v]) => encodeURIComponent(k) + '=' + encodeURIComponent(String(v)))
      .join('&');
  }

  /** Format any timestamp for display in Asia/Shanghai (北京时间). */
  function formatBeijingTime(value) {
    if (value == null || value === '') return '—';
    const text = String(value).trim();
    const hasTz = /[zZ]|[+-]\d{2}:?\d{2}$/.test(text);
    const spaceNaive = text.match(/^(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2})/);
    // New backend stamps are already Beijing wall clock (space-separated, no TZ).
    if (spaceNaive && !hasTz) {
      return spaceNaive[1] + ' ' + spaceNaive[2];
    }
    let date;
    if (!hasTz && /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}/.test(text)) {
      // Legacy fleet heartbeats were UTC naive ISO.
      date = new Date(text.endsWith('Z') ? text : (text + 'Z'));
    } else {
      date = new Date(text);
    }
    if (Number.isNaN(date.getTime())) return text;
    const parts = new Intl.DateTimeFormat('en-CA', {
      timeZone: 'Asia/Shanghai',
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      hour12: false
    }).formatToParts(date);
    const get = (type) => (parts.find((p) => p.type === type) || {}).value || '';
    return (
      get('year') + '-' + get('month') + '-' + get('day') +
      ' ' + get('hour') + ':' + get('minute') + ':' + get('second')
    );
  }

  window.formatBeijingTime = formatBeijingTime;

  window.MemStrataApi = {
    health: () => request('GET', '/api/health'),
    samples: (params) => request('GET', '/api/samples?' + q(params || {})),
    sampleDetail: (dataset, movieId) =>
      request('GET', '/api/sample-detail?' + q({ dataset: dataset, movie_id: movieId })),
    stageInspect: (dataset, movieId, stage) =>
      request('GET', '/api/review/stage?' + q({ dataset: dataset, movie_id: movieId, stage: stage })),
    reviewS3: (dataset, movieId) =>
      request('GET', '/api/review/s3?' + q({ dataset: dataset, movie_id: movieId })),
    reviewS4: (dataset, movieId) =>
      request('GET', '/api/review/s4?' + q({ dataset: dataset, movie_id: movieId })),
    draftS4: (body) => request('POST', '/api/review/s4/draft', body),
    applyS4: (body) => request('POST', '/api/review/s4/apply', body),
    acceptAllS4: (body) => request('POST', '/api/review/s4/accept-all', body),
    reviewS6: (dataset, movieId) =>
      request('GET', '/api/review/s6?' + q({ dataset: dataset, movie_id: movieId })),
    reviewS6Alts: (dataset, movieId, representationId) =>
      request(
        'GET',
        '/api/review/s6/alts?' +
          q({ dataset: dataset, movie_id: movieId, representation_id: representationId })
      ),
    applyS6: (body) => request('POST', '/api/review/s6/apply', body),
    continueReview: (body) => request('POST', '/api/review/continue', body),
    listJobs: () => request('GET', '/api/jobs'),
    listActiveJobs: () => request('GET', '/api/jobs/active'),
    getJob: (jobId) => request('GET', '/api/jobs/' + encodeURIComponent(jobId)),
    createJob: (body) => request('POST', '/api/jobs', body),
    stopJob: (jobId) => request('POST', '/api/jobs/' + encodeURIComponent(jobId) + '/stop', {}),
    stopAllJobs: () => request('POST', '/api/jobs/stop-all', {}),
    stopSampleJob: (dataset, movieId) =>
      request('POST', '/api/jobs/stop-sample', { dataset: dataset, movie_id: movieId }),
    jobLog: (jobId) => request('GET', '/api/job-log/' + encodeURIComponent(jobId)),
    fleet: (probe) => request('GET', '/api/fleet?' + q({ probe: probe ? '1' : '' }))
  };
})();
