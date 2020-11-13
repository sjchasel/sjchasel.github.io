/**
 * Copyright 2016 Google Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/

// DO NOT EDIT THIS GENERATED OUTPUT DIRECTLY!
// This file should be overwritten as part of your build process.
// If you need to extend the behavior of the generated service worker, the best approach is to write
// additional code and include it using the importScripts option:
//   https://github.com/GoogleChrome/sw-precache#importscripts-arraystring
//
// Alternatively, it's possible to make changes to the underlying template file and then use that as the
// new base for generating output, via the templateFilePath option:
//   https://github.com/GoogleChrome/sw-precache#templatefilepath-string
//
// If you go that route, make sure that whenever you update your sw-precache dependency, you reconcile any
// changes made to this original template file with your modified copy.

// This generated service worker JavaScript will precache your site's resources.
// The code needs to be saved in a .js file at the top-level of your site, and registered
// from your pages in order to be used. See
// https://github.com/googlechrome/sw-precache/blob/master/demo/app/js/service-worker-registration.js
// for an example of how you can register this script and handle various service worker events.

/* eslint-env worker, serviceworker */
/* eslint-disable indent, no-unused-vars, no-multiple-empty-lines, max-nested-callbacks, space-before-function-paren, quotes, comma-spacing */
'use strict';

var precacheConfig = [["E:/GitHubBlog/public/2020/01/28/hello-world/index.html","7fbf82897d9a2dd17bf0d1124f40c571"],["E:/GitHubBlog/public/2020/02/08/python爬虫学习总结/index.html","bed7c249e570c582b787b5ba414aa8c7"],["E:/GitHubBlog/public/2020/02/19/数据结构第一周-绪论/index.html","d533ae31dfc69bdd2c79175daee539fc"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-1/index.html","02169fce8cd61a9e8a69bcbc7343d9d8"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-2-创建数据集/index.html","8457d7bf54f2f582fe13467449ca4aa0"],["E:/GitHubBlog/public/2020/03/01/《R语言实战》学习总结-3-图形初阶/index.html","d3b9678c9bc7e7ea278529faaa28f459"],["E:/GitHubBlog/public/2020/03/02/数据结构第二周-线性结构1/index.html","bbb6cc3a0df1ff8deb33b118a2ae3ccd"],["E:/GitHubBlog/public/2020/03/03/《R语言实战》学习总结-4-基本数据管理/index.html","50c73cb803df8d265885433d19f121ec"],["E:/GitHubBlog/public/2020/03/14/数据结构第三周-线性结构2/index.html","aeda7e36968e409b5fc4326537dd8905"],["E:/GitHubBlog/public/2020/03/14/数据结构第四周-栈与队列/index.html","03d5b0d295d0e5dabd8cbcedd5eaec87"],["E:/GitHubBlog/public/2020/03/17/leetcode记录-1、88、20/index.html","31996e1400dd55d01eb6632f45e02208"],["E:/GitHubBlog/public/2020/03/20/温度转换程序/index.html","82579330b3cc27c97171c60ca274adc9"],["E:/GitHubBlog/public/2020/03/22/数据结构第五周-递归/index.html","786b2e8352adf959ac9cfb4c0015d92a"],["E:/GitHubBlog/public/2020/03/22/数据结构第六周-串与数组/index.html","51e2066b2f7b49648bf8a1bbe8fbe084"],["E:/GitHubBlog/public/2020/04/02/leetcode记录-52/index.html","49b4789c419ecb2b3fd9d7b1f2982380"],["E:/GitHubBlog/public/2020/04/17/Android第八周作业-每天第一次打开应用更新数据/index.html","04564c47f68da48244e69c514858fd0f"],["E:/GitHubBlog/public/2020/05/21/attention模型/index.html","aea096decaaa9d7a3d0660f8961e5a21"],["E:/GitHubBlog/public/2020/05/21/编码器与解码器框架/index.html","b78c4618608f899403bdd22110cedb05"],["E:/GitHubBlog/public/2020/07/11/思考，快与慢/index.html","19b6544322f064538ba2077ef9ac4c0f"],["E:/GitHubBlog/public/2020/07/13/20200713-0717总结/index.html","5c4b8abac846ab708430adcbc902313e"],["E:/GitHubBlog/public/2020/07/13/ZeroShotCeres Zero-Shot Relation Extraction/index.html","3cc506dd491514b66ded885c97cb5cde"],["E:/GitHubBlog/public/2020/07/20/Keyphrase Generation A Text Summarization Struggle/index.html","71a8b2128cdf31b3ab11c584d5972956"],["E:/GitHubBlog/public/2020/07/20/TPR/index.html","92de965e8b4d0367782d5060bd400299"],["E:/GitHubBlog/public/2020/07/23/Keyphrase Generation for Scientific Document Retrieval/index.html","d6c50ba5b9217afe083db27bd8759d11"],["E:/GitHubBlog/public/2020/07/24/20200718-0724总结/index.html","a4f7a1b80c38a564708f4f7f675d1914"],["E:/GitHubBlog/public/2020/07/31/20200725-0731总结/index.html","c77d137d3b656877e81b2d66ff4ef456"],["E:/GitHubBlog/public/2020/08/07/20200801-0807总结/index.html","db5c607e994003787e599288a74f684e"],["E:/GitHubBlog/public/2020/08/12/逻辑回归算法/index.html","f5f1397c65bc6b03aa249ff1d0df6018"],["E:/GitHubBlog/public/2020/08/13/李宏毅-机器学习21/index.html","9670d4f63fc7b95c1acbbc387505ef54"],["E:/GitHubBlog/public/2020/08/14/20200808-0815总结/index.html","00b62f45fefac02f3d27d9068539edde"],["E:/GitHubBlog/public/2020/09/03/20200816-0903总结/index.html","c6f10f5bb86156a7ae36af43d522e491"],["E:/GitHubBlog/public/2020/09/16/20200914-0918总结/index.html","d1e3202ec125e309da4434e804b46118"],["E:/GitHubBlog/public/2020/09/25/20200919-0925总结/index.html","f5337ad9d387bf4bdbbe8c7e9c001eee"],["E:/GitHubBlog/public/2020/10/06/《Deep Keyphrase Generation》详细解析！/index.html","fa5b3b95d67f247d17dcebfb34b4ae78"],["E:/GitHubBlog/public/2020/10/16/20201008-1017总结/index.html","ad1007b726c64aab5261339190f688e3"],["E:/GitHubBlog/public/2020/10/20/特征选择的几种方法(一)————Filter过滤法/index.html","9d1061c12930efd8229e210bdf31fe21"],["E:/GitHubBlog/public/2020/10/21/PCA/index.html","7d5e19e62b9e1e25bdc9dfe03a9c27ff"],["E:/GitHubBlog/public/2020/10/22/20201018-1023总结/index.html","6ac6c867f4b7bcffcf2c179a6a854673"],["E:/GitHubBlog/public/2020/10/24/20201024-1030/index.html","9f0129a09cff93d63d7d02948628c1ca"],["E:/GitHubBlog/public/2020/10/30/1030讨论记录/index.html","507ce016e4de0e5e55778742e15a3136"],["E:/GitHubBlog/public/2020/11/02/《如何写一个商业计划书》读后感/index.html","b8b9c4c1f3653b76e664ca84f43c348d"],["E:/GitHubBlog/public/2020/11/03/20201103-总结/index.html","6ab879de8f1f89245bce763ddcf355ec"],["E:/GitHubBlog/public/2020/11/04/《思考，快与慢》读后感/index.html","08a99d666a2ad6039a1b3df48f9dff68"],["E:/GitHubBlog/public/2020/11/05/优化方法!!——一些前置知识/index.html","6e1cc2b6676e4a51db739a227aeccb47"],["E:/GitHubBlog/public/2020/11/07/决策树/index.html","3c20428db521332a9fe2d15f0032873b"],["E:/GitHubBlog/public/2020/11/13/20201107-1113总结/index.html","9996d0e2e92b477f83f80e09e734bb23"],["E:/GitHubBlog/public/archives/2020/01/index.html","452fc6fc2a84f43a21181730f6de817b"],["E:/GitHubBlog/public/archives/2020/02/index.html","0e0e49d92706fb9ac7b69672d7cdd1ba"],["E:/GitHubBlog/public/archives/2020/03/index.html","191b3bb8b577871bd10a5395ca7fdb15"],["E:/GitHubBlog/public/archives/2020/04/index.html","8bbb6e3beb259741d8ee7f50b3524aae"],["E:/GitHubBlog/public/archives/2020/05/index.html","0165171c9a086283f75767339331baca"],["E:/GitHubBlog/public/archives/2020/07/index.html","d3b4a1ca62c1c7a5a6393a824a7399de"],["E:/GitHubBlog/public/archives/2020/08/index.html","7ff8f8baf64c421a4a09b44bb3f46e0d"],["E:/GitHubBlog/public/archives/2020/09/index.html","26a586cbcfcc3c1345987e6c6d5b9073"],["E:/GitHubBlog/public/archives/2020/10/index.html","01216399f5f47fd0390687f3946c00fc"],["E:/GitHubBlog/public/archives/2020/11/index.html","19dda3688e20f96cb880d5c8f6e4e423"],["E:/GitHubBlog/public/archives/2020/index.html","0aba122581b2f1554d1919160f90909c"],["E:/GitHubBlog/public/archives/2020/page/2/index.html","fce4fe3605184128351915fec166f362"],["E:/GitHubBlog/public/archives/2020/page/3/index.html","cf0ddc1d2ecbf62bfa886637a4038f3e"],["E:/GitHubBlog/public/archives/2020/page/4/index.html","010a11b255af479b64c0b9071dd251d4"],["E:/GitHubBlog/public/archives/2020/page/5/index.html","31ac3f9ce34ab5ff06484b482c45e1fe"],["E:/GitHubBlog/public/archives/index.html","033955c2f324bc61a8c49f9f1f4dc223"],["E:/GitHubBlog/public/archives/page/2/index.html","bd334b06e6d60284cd0de33bbbf324e2"],["E:/GitHubBlog/public/archives/page/3/index.html","2868b7de545999e575625d9796f9ad5c"],["E:/GitHubBlog/public/archives/page/4/index.html","2e4daaf3197b5f227f8c9805c7384f73"],["E:/GitHubBlog/public/archives/page/5/index.html","51765ce2716507d52e64f05b83e826d8"],["E:/GitHubBlog/public/css/404.css","b1bb50e1fabe41adcec483f6427fb3aa"],["E:/GitHubBlog/public/css/index.css","2389364df340948a5b822422956c2c87"],["E:/GitHubBlog/public/css/var.css","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/img/algolia.svg","fd40b88ac5370a5353a50b8175c1f367"],["E:/GitHubBlog/public/img/avatar.png","6cc4a809d23e3d8946a299ae4ce4e4cd"],["E:/GitHubBlog/public/index.html","f402524393dd69cd299a040342e0fe6b"],["E:/GitHubBlog/public/js/copy.js","45aae54bf2e43ac27ecc41eb5e0bacf7"],["E:/GitHubBlog/public/js/fancybox.js","f84d626654b9bbc05720952b3effe062"],["E:/GitHubBlog/public/js/fireworks.js","35933ce61c158ef9c33b5e089fe757ac"],["E:/GitHubBlog/public/js/head.js","347edd99f8e3921b45fa617e839d8182"],["E:/GitHubBlog/public/js/hexo-theme-melody.js","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/js/katex.js","d971ba8b8dee1120ef66744b89cfd2b1"],["E:/GitHubBlog/public/js/scroll.js","a12ad7e37b9e325f8da3d0523d3757c7"],["E:/GitHubBlog/public/js/search/algolia.js","53160985d32d6fd66d98aa0e05b081ac"],["E:/GitHubBlog/public/js/search/local-search.js","1565b508bd866ed6fbd724a3858af5d8"],["E:/GitHubBlog/public/js/sidebar.js","d24db780974e661198eeb2e45f20a28f"],["E:/GitHubBlog/public/js/third-party/anime.min.js","9b4bbe6deb700e1c3606eab732f5eea5"],["E:/GitHubBlog/public/js/third-party/canvas-ribbon.js","09c181544ddff1db701db02ac30453e0"],["E:/GitHubBlog/public/js/third-party/jquery.fancybox.min.js","3c9fa1c1199cd4f874d855ecb1641335"],["E:/GitHubBlog/public/js/third-party/jquery.min.js","c9f5aeeca3ad37bf2aa006139b935f0a"],["E:/GitHubBlog/public/js/third-party/reveal/head.min.js","aad121203010122e05f1766d92385214"],["E:/GitHubBlog/public/js/third-party/reveal/highlight.min.js","44594243bec43813a16371af8fe7e105"],["E:/GitHubBlog/public/js/third-party/reveal/markdown.min.js","7ec4cef5a7fe3f0bf0eb4dc6d7bca114"],["E:/GitHubBlog/public/js/third-party/reveal/marked.min.js","c2a88705e206d71dc21fdc4445349127"],["E:/GitHubBlog/public/js/third-party/reveal/math.min.js","0a278fee2e57c530ab07f7d2d9ea8d96"],["E:/GitHubBlog/public/js/third-party/reveal/notes.min.js","89a0dfae4d706f9c75b317f686c3aa14"],["E:/GitHubBlog/public/js/third-party/reveal/reveal.min.js","8988419d67efb5fe93e291a357e26ec9"],["E:/GitHubBlog/public/js/third-party/reveal/zoom.min.js","9791f96e63e7d534cba2b67d4bda0419"],["E:/GitHubBlog/public/js/third-party/velocity.min.js","64da069aba987ea0512cf610600a56d1"],["E:/GitHubBlog/public/js/third-party/velocity.ui.min.js","c8ca438424a080620f7b2f4ee4b0fff1"],["E:/GitHubBlog/public/js/transition.js","911db4268f0f6621073afcced9e1bfef"],["E:/GitHubBlog/public/js/utils.js","3ff3423d966a1c351e9867813b3f6d36"],["E:/GitHubBlog/public/page/2/index.html","405de5bf82d168cfe7fc54d4304a2802"],["E:/GitHubBlog/public/page/3/index.html","7ac5e7fe4398ed629b906690207e247e"],["E:/GitHubBlog/public/page/4/index.html","6893793bfc078e29efe3256f843f431b"],["E:/GitHubBlog/public/page/5/index.html","7d130040803baa31e194403441f64f36"],["E:/GitHubBlog/public/tags/Android/index.html","012f3b7bbd0d74d350cdea5878449994"],["E:/GitHubBlog/public/tags/NLP/index.html","478f2c0c06bd39b0e1fade1da083b274"],["E:/GitHubBlog/public/tags/NLP/page/2/index.html","29ea85c994ea6fe8e9de7fe6bcae25f7"],["E:/GitHubBlog/public/tags/R/index.html","76dc7f98d6929c744b2c70562cc693f4"],["E:/GitHubBlog/public/tags/index.html","9a3ad796ffeef5deaeeb72f2be5e2e6a"],["E:/GitHubBlog/public/tags/java/index.html","5698727ca40dc3bfc3b9858eccfe6b2f"],["E:/GitHubBlog/public/tags/leetcode/index.html","c9fad743c114989e52414c73c5203e1b"],["E:/GitHubBlog/public/tags/python/index.html","512ab276cecc13135b74caca58a00630"],["E:/GitHubBlog/public/tags/优化方法/index.html","954e3fb57f5be3985bdf5fa66b74bac2"],["E:/GitHubBlog/public/tags/总结/index.html","44452fab156efa16e0176f69be22fc36"],["E:/GitHubBlog/public/tags/总结/page/2/index.html","ae6cd1286878ecdabb83be65c7e8d843"],["E:/GitHubBlog/public/tags/数学/index.html","890126a10e56db1b70cd1cfe0633f17a"],["E:/GitHubBlog/public/tags/数据分析/index.html","9fa2f517783471b1f7f3cf6d5b053182"],["E:/GitHubBlog/public/tags/数据挖掘/index.html","05aee572147f9137151688c2e72bb2a7"],["E:/GitHubBlog/public/tags/数据结构/index.html","d5d0e8b5bff2aabe04b65645a595ac61"],["E:/GitHubBlog/public/tags/机器学习/index.html","d15a0cbbcf96c7aa4da5ac98c3c76d98"],["E:/GitHubBlog/public/tags/深度学习/index.html","5579b5553c6a386477ab1e769af9f327"],["E:/GitHubBlog/public/tags/爬虫/index.html","0b0b5b6cfb6bdea4dbd3e121f56819bd"],["E:/GitHubBlog/public/tags/笔记/index.html","856f07f4b7d5ed26aaecdd6cb6f2b711"],["E:/GitHubBlog/public/tags/论文/index.html","fded7ca84e936b8ee03325c0200e5993"],["E:/GitHubBlog/public/tags/论文/page/2/index.html","9d1b677d149dd8571c3aef28f2f76aa0"],["E:/GitHubBlog/public/tags/读书笔记/index.html","33c3feb1c277a9175b3d5c8af09ec5e4"]];
var cacheName = 'sw-precache-v3--' + (self.registration ? self.registration.scope : '');


var ignoreUrlParametersMatching = [/^utm_/];



var addDirectoryIndex = function(originalUrl, index) {
    var url = new URL(originalUrl);
    if (url.pathname.slice(-1) === '/') {
      url.pathname += index;
    }
    return url.toString();
  };

var cleanResponse = function(originalResponse) {
    // If this is not a redirected response, then we don't have to do anything.
    if (!originalResponse.redirected) {
      return Promise.resolve(originalResponse);
    }

    // Firefox 50 and below doesn't support the Response.body stream, so we may
    // need to read the entire body to memory as a Blob.
    var bodyPromise = 'body' in originalResponse ?
      Promise.resolve(originalResponse.body) :
      originalResponse.blob();

    return bodyPromise.then(function(body) {
      // new Response() is happy when passed either a stream or a Blob.
      return new Response(body, {
        headers: originalResponse.headers,
        status: originalResponse.status,
        statusText: originalResponse.statusText
      });
    });
  };

var createCacheKey = function(originalUrl, paramName, paramValue,
                           dontCacheBustUrlsMatching) {
    // Create a new URL object to avoid modifying originalUrl.
    var url = new URL(originalUrl);

    // If dontCacheBustUrlsMatching is not set, or if we don't have a match,
    // then add in the extra cache-busting URL parameter.
    if (!dontCacheBustUrlsMatching ||
        !(url.pathname.match(dontCacheBustUrlsMatching))) {
      url.search += (url.search ? '&' : '') +
        encodeURIComponent(paramName) + '=' + encodeURIComponent(paramValue);
    }

    return url.toString();
  };

var isPathWhitelisted = function(whitelist, absoluteUrlString) {
    // If the whitelist is empty, then consider all URLs to be whitelisted.
    if (whitelist.length === 0) {
      return true;
    }

    // Otherwise compare each path regex to the path of the URL passed in.
    var path = (new URL(absoluteUrlString)).pathname;
    return whitelist.some(function(whitelistedPathRegex) {
      return path.match(whitelistedPathRegex);
    });
  };

var stripIgnoredUrlParameters = function(originalUrl,
    ignoreUrlParametersMatching) {
    var url = new URL(originalUrl);
    // Remove the hash; see https://github.com/GoogleChrome/sw-precache/issues/290
    url.hash = '';

    url.search = url.search.slice(1) // Exclude initial '?'
      .split('&') // Split into an array of 'key=value' strings
      .map(function(kv) {
        return kv.split('='); // Split each 'key=value' string into a [key, value] array
      })
      .filter(function(kv) {
        return ignoreUrlParametersMatching.every(function(ignoredRegex) {
          return !ignoredRegex.test(kv[0]); // Return true iff the key doesn't match any of the regexes.
        });
      })
      .map(function(kv) {
        return kv.join('='); // Join each [key, value] array into a 'key=value' string
      })
      .join('&'); // Join the array of 'key=value' strings into a string with '&' in between each

    return url.toString();
  };


var hashParamName = '_sw-precache';
var urlsToCacheKeys = new Map(
  precacheConfig.map(function(item) {
    var relativeUrl = item[0];
    var hash = item[1];
    var absoluteUrl = new URL(relativeUrl, self.location);
    var cacheKey = createCacheKey(absoluteUrl, hashParamName, hash, false);
    return [absoluteUrl.toString(), cacheKey];
  })
);

function setOfCachedUrls(cache) {
  return cache.keys().then(function(requests) {
    return requests.map(function(request) {
      return request.url;
    });
  }).then(function(urls) {
    return new Set(urls);
  });
}

self.addEventListener('install', function(event) {
  event.waitUntil(
    caches.open(cacheName).then(function(cache) {
      return setOfCachedUrls(cache).then(function(cachedUrls) {
        return Promise.all(
          Array.from(urlsToCacheKeys.values()).map(function(cacheKey) {
            // If we don't have a key matching url in the cache already, add it.
            if (!cachedUrls.has(cacheKey)) {
              var request = new Request(cacheKey, {credentials: 'same-origin'});
              return fetch(request).then(function(response) {
                // Bail out of installation unless we get back a 200 OK for
                // every request.
                if (!response.ok) {
                  throw new Error('Request for ' + cacheKey + ' returned a ' +
                    'response with status ' + response.status);
                }

                return cleanResponse(response).then(function(responseToCache) {
                  return cache.put(cacheKey, responseToCache);
                });
              });
            }
          })
        );
      });
    }).then(function() {
      
      // Force the SW to transition from installing -> active state
      return self.skipWaiting();
      
    })
  );
});

self.addEventListener('activate', function(event) {
  var setOfExpectedUrls = new Set(urlsToCacheKeys.values());

  event.waitUntil(
    caches.open(cacheName).then(function(cache) {
      return cache.keys().then(function(existingRequests) {
        return Promise.all(
          existingRequests.map(function(existingRequest) {
            if (!setOfExpectedUrls.has(existingRequest.url)) {
              return cache.delete(existingRequest);
            }
          })
        );
      });
    }).then(function() {
      
      return self.clients.claim();
      
    })
  );
});


self.addEventListener('fetch', function(event) {
  if (event.request.method === 'GET') {
    // Should we call event.respondWith() inside this fetch event handler?
    // This needs to be determined synchronously, which will give other fetch
    // handlers a chance to handle the request if need be.
    var shouldRespond;

    // First, remove all the ignored parameters and hash fragment, and see if we
    // have that URL in our cache. If so, great! shouldRespond will be true.
    var url = stripIgnoredUrlParameters(event.request.url, ignoreUrlParametersMatching);
    shouldRespond = urlsToCacheKeys.has(url);

    // If shouldRespond is false, check again, this time with 'index.html'
    // (or whatever the directoryIndex option is set to) at the end.
    var directoryIndex = 'index.html';
    if (!shouldRespond && directoryIndex) {
      url = addDirectoryIndex(url, directoryIndex);
      shouldRespond = urlsToCacheKeys.has(url);
    }

    // If shouldRespond is still false, check to see if this is a navigation
    // request, and if so, whether the URL matches navigateFallbackWhitelist.
    var navigateFallback = '';
    if (!shouldRespond &&
        navigateFallback &&
        (event.request.mode === 'navigate') &&
        isPathWhitelisted([], event.request.url)) {
      url = new URL(navigateFallback, self.location).toString();
      shouldRespond = urlsToCacheKeys.has(url);
    }

    // If shouldRespond was set to true at any point, then call
    // event.respondWith(), using the appropriate cache key.
    if (shouldRespond) {
      event.respondWith(
        caches.open(cacheName).then(function(cache) {
          return cache.match(urlsToCacheKeys.get(url)).then(function(response) {
            if (response) {
              return response;
            }
            throw Error('The cached response that was expected is missing.');
          });
        }).catch(function(e) {
          // Fall back to just fetch()ing the request if some unexpected error
          // prevented the cached response from being valid.
          console.warn('Couldn\'t serve response for "%s" from cache: %O', event.request.url, e);
          return fetch(event.request);
        })
      );
    }
  }
});







