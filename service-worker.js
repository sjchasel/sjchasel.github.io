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

var precacheConfig = [["E:/GitHubBlog/public/2020/01/28/hello-world/index.html","5ad0f92d768589d548d13090feb0d99f"],["E:/GitHubBlog/public/2020/02/08/python爬虫学习总结/index.html","fe963de438c87bba778bb5470e830471"],["E:/GitHubBlog/public/2020/02/19/数据结构第一周-绪论/index.html","ac8a8ef00f30da4fe134adc314beec2a"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-1/index.html","defd4383154362286bd1d067e8eefcd4"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-2-创建数据集/index.html","15084d92d3fa208e917ac555c92d65a9"],["E:/GitHubBlog/public/2020/03/01/《R语言实战》学习总结-3-图形初阶/index.html","1b2e6e82d822babff6ae123346380d97"],["E:/GitHubBlog/public/2020/03/02/数据结构第二周-线性结构1/index.html","7f35dc19ea72facd23c17610a829c266"],["E:/GitHubBlog/public/2020/03/03/《R语言实战》学习总结-4-基本数据管理/index.html","0b2551d10b69a4ca8d9f01aea98b96df"],["E:/GitHubBlog/public/2020/03/14/数据结构第三周-线性结构2/index.html","76be260855a764eef10dd8b9c7ef32d4"],["E:/GitHubBlog/public/2020/03/14/数据结构第四周-栈与队列/index.html","f9297a5e62a4995df5ded478bf275eb5"],["E:/GitHubBlog/public/2020/03/17/leetcode记录-1、88、20/index.html","c5b048a6758b42bc2842c9099ebdaafc"],["E:/GitHubBlog/public/2020/03/20/温度转换程序/index.html","0172e5a67637efea6a82c5e0769f9908"],["E:/GitHubBlog/public/2020/03/22/数据结构第五周-递归/index.html","91e960df471b0cf5ad08a1a358bbf42c"],["E:/GitHubBlog/public/2020/03/22/数据结构第六周-串与数组/index.html","623183bbce6ffe06c5debb4efc3612da"],["E:/GitHubBlog/public/2020/04/02/leetcode记录-52/index.html","30fd890ffbbc87eb8849634382776c43"],["E:/GitHubBlog/public/2020/04/17/Android第八周作业-每天第一次打开应用更新数据/index.html","0037c0a6297ced2b64b4329a3fcd5596"],["E:/GitHubBlog/public/2020/05/21/attention模型/index.html","4a61d0b51d57e4405809b1c6cc131d37"],["E:/GitHubBlog/public/2020/05/21/编码器与解码器框架/index.html","0b24b0a19e06c1dca07df3385c091f25"],["E:/GitHubBlog/public/2020/07/11/思考，快与慢/index.html","6a8cb3d9329521e39c855d01bbb6725e"],["E:/GitHubBlog/public/2020/07/13/20200713-0717总结/index.html","af1ca7042c99889da82689dab66a65a8"],["E:/GitHubBlog/public/2020/07/13/ZeroShotCeres Zero-Shot Relation Extraction/index.html","88dd20c68e834fe1972b16a3cac1e454"],["E:/GitHubBlog/public/2020/07/20/Keyphrase Generation A Text Summarization Struggle/index.html","9776458030d759f48b342c50744ebb47"],["E:/GitHubBlog/public/2020/07/20/TPR/index.html","a0d4fb6fc4cf727442d94fb6eb189771"],["E:/GitHubBlog/public/2020/07/23/Keyphrase Generation for Scientific Document Retrieval/index.html","4c3b76c106c1c2331e2355e9302f94af"],["E:/GitHubBlog/public/2020/07/24/20200718-0724总结/index.html","57caa6f80465cce6078c49f2a9df80d0"],["E:/GitHubBlog/public/2020/07/31/20200725-0731总结/index.html","f4bd8758755d85693cdde478ff5a0b57"],["E:/GitHubBlog/public/2020/08/07/20200801-0807总结/index.html","3fd31fc7328749d5c56063fba00ed6fa"],["E:/GitHubBlog/public/2020/08/12/逻辑回归算法/index.html","50192a1417545e53f6560bcb71ff7736"],["E:/GitHubBlog/public/2020/08/13/李宏毅-机器学习21/index.html","c393abeffe5910728720c8fbfc191fd2"],["E:/GitHubBlog/public/2020/08/14/20200808-0815总结/index.html","dab0e5d6f1585f7ea14d5c1463bdfbfc"],["E:/GitHubBlog/public/archives/2020/01/index.html","56835c45f3c526bdfc85c342a871edaf"],["E:/GitHubBlog/public/archives/2020/02/index.html","61c80aea6acf460816cb53653dac0fb7"],["E:/GitHubBlog/public/archives/2020/03/index.html","af0ac10fc3b2026852e997322ddf9c2a"],["E:/GitHubBlog/public/archives/2020/04/index.html","e0e7b2235ce3bccb6b6590f39ca33ab4"],["E:/GitHubBlog/public/archives/2020/05/index.html","f1786e5b38b6eee31d0f143b7e5b4bce"],["E:/GitHubBlog/public/archives/2020/07/index.html","d2a7fe10911c330ef8518d31f327bc62"],["E:/GitHubBlog/public/archives/2020/08/index.html","e1735a76589f2fea33065e43b8bd93f2"],["E:/GitHubBlog/public/archives/2020/index.html","33eae5e41c9a162cd34688e1b7dc5e25"],["E:/GitHubBlog/public/archives/2020/page/2/index.html","99773ae05fcaeb7aaf6aa32d04614d9f"],["E:/GitHubBlog/public/archives/2020/page/3/index.html","c9d4af2379f5be62ac3083a92c2d8381"],["E:/GitHubBlog/public/archives/index.html","ca32966548f0a15636907ddef03ffe78"],["E:/GitHubBlog/public/archives/page/2/index.html","196054cc20406c6ac1117d8055bb08d2"],["E:/GitHubBlog/public/archives/page/3/index.html","e7249860e2ecf45c5f23479400307b25"],["E:/GitHubBlog/public/css/404.css","b1bb50e1fabe41adcec483f6427fb3aa"],["E:/GitHubBlog/public/css/index.css","2389364df340948a5b822422956c2c87"],["E:/GitHubBlog/public/css/var.css","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/img/algolia.svg","fd40b88ac5370a5353a50b8175c1f367"],["E:/GitHubBlog/public/img/avatar.png","6cc4a809d23e3d8946a299ae4ce4e4cd"],["E:/GitHubBlog/public/index.html","c3195d8c6297dbc8ded7a9847e068f91"],["E:/GitHubBlog/public/js/copy.js","45aae54bf2e43ac27ecc41eb5e0bacf7"],["E:/GitHubBlog/public/js/fancybox.js","f84d626654b9bbc05720952b3effe062"],["E:/GitHubBlog/public/js/fireworks.js","35933ce61c158ef9c33b5e089fe757ac"],["E:/GitHubBlog/public/js/head.js","347edd99f8e3921b45fa617e839d8182"],["E:/GitHubBlog/public/js/hexo-theme-melody.js","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/js/katex.js","d971ba8b8dee1120ef66744b89cfd2b1"],["E:/GitHubBlog/public/js/scroll.js","a12ad7e37b9e325f8da3d0523d3757c7"],["E:/GitHubBlog/public/js/search/algolia.js","53160985d32d6fd66d98aa0e05b081ac"],["E:/GitHubBlog/public/js/search/local-search.js","1565b508bd866ed6fbd724a3858af5d8"],["E:/GitHubBlog/public/js/sidebar.js","d24db780974e661198eeb2e45f20a28f"],["E:/GitHubBlog/public/js/third-party/anime.min.js","9b4bbe6deb700e1c3606eab732f5eea5"],["E:/GitHubBlog/public/js/third-party/canvas-ribbon.js","09c181544ddff1db701db02ac30453e0"],["E:/GitHubBlog/public/js/third-party/jquery.fancybox.min.js","3c9fa1c1199cd4f874d855ecb1641335"],["E:/GitHubBlog/public/js/third-party/jquery.min.js","c9f5aeeca3ad37bf2aa006139b935f0a"],["E:/GitHubBlog/public/js/third-party/reveal/head.min.js","aad121203010122e05f1766d92385214"],["E:/GitHubBlog/public/js/third-party/reveal/highlight.min.js","44594243bec43813a16371af8fe7e105"],["E:/GitHubBlog/public/js/third-party/reveal/markdown.min.js","7ec4cef5a7fe3f0bf0eb4dc6d7bca114"],["E:/GitHubBlog/public/js/third-party/reveal/marked.min.js","c2a88705e206d71dc21fdc4445349127"],["E:/GitHubBlog/public/js/third-party/reveal/math.min.js","0a278fee2e57c530ab07f7d2d9ea8d96"],["E:/GitHubBlog/public/js/third-party/reveal/notes.min.js","89a0dfae4d706f9c75b317f686c3aa14"],["E:/GitHubBlog/public/js/third-party/reveal/reveal.min.js","8988419d67efb5fe93e291a357e26ec9"],["E:/GitHubBlog/public/js/third-party/reveal/zoom.min.js","9791f96e63e7d534cba2b67d4bda0419"],["E:/GitHubBlog/public/js/third-party/velocity.min.js","64da069aba987ea0512cf610600a56d1"],["E:/GitHubBlog/public/js/third-party/velocity.ui.min.js","c8ca438424a080620f7b2f4ee4b0fff1"],["E:/GitHubBlog/public/js/transition.js","911db4268f0f6621073afcced9e1bfef"],["E:/GitHubBlog/public/js/utils.js","3ff3423d966a1c351e9867813b3f6d36"],["E:/GitHubBlog/public/page/2/index.html","8162c31d6e26d1e29e958942ba89aabd"],["E:/GitHubBlog/public/page/3/index.html","48a50ac0e6ecd8ba1bb8592cc08fb37d"],["E:/GitHubBlog/public/tags/Android/index.html","e6c4e4670826ca2a259a29539138d196"],["E:/GitHubBlog/public/tags/NLP/index.html","42d3698bbc7b0e700dcfd04f5ad21cc2"],["E:/GitHubBlog/public/tags/R/index.html","0bf765d5750892722fffb1cd05a66347"],["E:/GitHubBlog/public/tags/index.html","296fcf1f240dc0f7bfc6dfb822ff8ab2"],["E:/GitHubBlog/public/tags/java/index.html","30d590dc7ee58ea732fc9b804aba235c"],["E:/GitHubBlog/public/tags/leetcode/index.html","6f762dc54a1ca056c1a007bc45adaaa4"],["E:/GitHubBlog/public/tags/python/index.html","d994bb15e7e163ddfd9ed9d5e7ac8ec6"],["E:/GitHubBlog/public/tags/总结/index.html","5c944df5af458fc6ef475b1b3d76b6de"],["E:/GitHubBlog/public/tags/数据分析/index.html","b5bce75a508974e60e53da4a13122054"],["E:/GitHubBlog/public/tags/数据结构/index.html","cab82fd74bf736f1b086a8a795fa824c"],["E:/GitHubBlog/public/tags/机器学习/index.html","45a2e7856fa6a1a9f93903f21218a75b"],["E:/GitHubBlog/public/tags/深度学习/index.html","1c2746b2306c5cc258240b85d11e6496"],["E:/GitHubBlog/public/tags/爬虫/index.html","5b51983e05434d0fedcc9c575c2317bf"],["E:/GitHubBlog/public/tags/笔记/index.html","3351e3f98b26d7f8c261b9e7c1243512"],["E:/GitHubBlog/public/tags/论文/index.html","e78299a9334398e6b45c24f18a1e6d66"],["E:/GitHubBlog/public/tags/论文/page/2/index.html","304de397ea5b2f13717bf223cdc2a5a5"],["E:/GitHubBlog/public/tags/读书笔记/index.html","458aff4621f135f7b0da8d3cfebb606d"]];
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







