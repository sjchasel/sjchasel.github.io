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

var precacheConfig = [["E:/GitHubBlog/public/2020/01/28/hello-world/index.html","bf017d8df23bd189f6553697791bf413"],["E:/GitHubBlog/public/2020/02/08/python爬虫学习总结/index.html","d63600f51f6e50aa32b902fb4f021613"],["E:/GitHubBlog/public/2020/02/19/数据结构第一周-绪论/index.html","bade15f0c788d7ec6bdb97bf76f382d1"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-1/index.html","16ee2f63cf58fd3c3c460ce2c1ed8808"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-2-创建数据集/index.html","8408a6576a8e5081d34aa6a92d48457a"],["E:/GitHubBlog/public/2020/03/01/《R语言实战》学习总结-3-图形初阶/index.html","72a3f08254fd5d22f3d8a93d1226fa7a"],["E:/GitHubBlog/public/2020/03/02/数据结构第二周-线性结构1/index.html","bad474b247dd9b82811e198990555176"],["E:/GitHubBlog/public/2020/03/03/《R语言实战》学习总结-4-基本数据管理/index.html","faf898395be0e85eed7dba157e44980c"],["E:/GitHubBlog/public/2020/03/14/数据结构第三周-线性结构2/index.html","ad7689dd6668bf9c171e268150b56585"],["E:/GitHubBlog/public/2020/03/14/数据结构第四周-栈与队列/index.html","e20f3e938918a2bf26fe369d13470be1"],["E:/GitHubBlog/public/2020/03/17/leetcode记录-1、88、20/index.html","2b8aaa392859ed851683957ab5b9ca17"],["E:/GitHubBlog/public/2020/03/20/温度转换程序/index.html","282ff5178264d388afcffc4620f44922"],["E:/GitHubBlog/public/2020/03/22/数据结构第五周-递归/index.html","8fe247d2e462d40911f687db200eb0cc"],["E:/GitHubBlog/public/2020/03/22/数据结构第六周-串与数组/index.html","ba3749147895beae7aeffb6de23c189c"],["E:/GitHubBlog/public/2020/04/02/leetcode记录-52/index.html","703ecfbe5dbe24c1d0432344da500cf9"],["E:/GitHubBlog/public/2020/04/17/Android第八周作业-每天第一次打开应用更新数据/index.html","1040062761835a994f1af0fed211ed6a"],["E:/GitHubBlog/public/2020/05/21/attention模型/index.html","08b896acac306bae529c10148af5691c"],["E:/GitHubBlog/public/2020/05/21/编码器与解码器框架/index.html","f3ee4b761528764984c5768d88725a48"],["E:/GitHubBlog/public/2020/07/11/思考，快与慢/index.html","0fe6b7c0715cd11d1414207fdb157d02"],["E:/GitHubBlog/public/2020/07/13/20200713-0717总结/index.html","75074d73e846ba69ef376017737ffba3"],["E:/GitHubBlog/public/2020/07/13/ZeroShotCeres Zero-Shot Relation Extraction/index.html","36234ad0a9e055a862d79f48350d04ab"],["E:/GitHubBlog/public/2020/07/20/Keyphrase Generation A Text Summarization Struggle/index.html","918a6cdedd62e74e49c75743adee0e7b"],["E:/GitHubBlog/public/2020/07/20/TPR/index.html","5662da4eb8c9e8f9da24a3b1cefbe02d"],["E:/GitHubBlog/public/2020/07/23/Keyphrase Generation for Scientific Document Retrieval/index.html","8173c6234bc4b3bec6230ca31c7f41ea"],["E:/GitHubBlog/public/2020/07/24/20200718-0724总结/index.html","b579e244f207cb257f29c5c38e1e57e5"],["E:/GitHubBlog/public/2020/07/31/20200725-0731总结/index.html","b8f9546236112454142f5b3dc9a9d6b5"],["E:/GitHubBlog/public/2020/08/07/20200801-0807总结/index.html","6458d8c6dbb1b717ddf2593540ad29df"],["E:/GitHubBlog/public/2020/08/12/逻辑回归算法/index.html","c0b0c51eefffdfe30c51dc353c1c6122"],["E:/GitHubBlog/public/2020/08/13/李宏毅-机器学习21/index.html","9f1c4596a90bb0cc1b6cef769a822def"],["E:/GitHubBlog/public/2020/08/14/20200808-0815总结/index.html","bf323a63589ffbcfe7224b17e57cc605"],["E:/GitHubBlog/public/2020/09/03/20200816-0903总结/index.html","23368eefb5d250d616c2eeb1494b6e39"],["E:/GitHubBlog/public/2020/09/16/20200914-0918总结/index.html","693084ed5f9478183e88e9bb6d364273"],["E:/GitHubBlog/public/2020/09/25/20200919-0925总结/index.html","4db7570092d08e73eedcdbdf91801896"],["E:/GitHubBlog/public/2020/10/06/《Deep Keyphrase Generation》详细解析！/index.html","7e68aa8655f84cbb37c94da03e7a62d9"],["E:/GitHubBlog/public/2020/10/16/20201008-1017总结/index.html","452d2e4b63f81f223c4efc753a574c57"],["E:/GitHubBlog/public/2020/10/20/特征选择/index.html","fdc269243541cc0146c17e171e45c0a5"],["E:/GitHubBlog/public/archives/2020/01/index.html","5eeeff92fc13186d4da841e9c4a4ce19"],["E:/GitHubBlog/public/archives/2020/02/index.html","b30771f05c99b878e5fa3794475879a6"],["E:/GitHubBlog/public/archives/2020/03/index.html","20c4654c089c92fd699867140bc5a119"],["E:/GitHubBlog/public/archives/2020/04/index.html","c26afdec484146644775f99f3c28c3b8"],["E:/GitHubBlog/public/archives/2020/05/index.html","af687c3fe3fdbde3b9a259601fde5c14"],["E:/GitHubBlog/public/archives/2020/07/index.html","0fb9ef8556ca2f912292c58f7f47d0a9"],["E:/GitHubBlog/public/archives/2020/08/index.html","38ead1ba186dea9d7e0aea7a5f8b59c0"],["E:/GitHubBlog/public/archives/2020/09/index.html","da7bf7fc664a3f29e6577779aea47959"],["E:/GitHubBlog/public/archives/2020/10/index.html","1efb68be056d4ae943b1f4113408de17"],["E:/GitHubBlog/public/archives/2020/index.html","91a145f361416d47c7265d3bdf8a2938"],["E:/GitHubBlog/public/archives/2020/page/2/index.html","3ff7f1b22fb61a5c118d9b12488eec4c"],["E:/GitHubBlog/public/archives/2020/page/3/index.html","bfbc4b6a799a1cff42dd388d4f404b73"],["E:/GitHubBlog/public/archives/2020/page/4/index.html","90f8dd05b0b25ad61c8954d263bf38c3"],["E:/GitHubBlog/public/archives/index.html","457f941eb1abb3413fed346798c21b0d"],["E:/GitHubBlog/public/archives/page/2/index.html","09a6bb5c1ea0644ca2aa6523810b79e3"],["E:/GitHubBlog/public/archives/page/3/index.html","c7c970e58a29c7e8aeb625805b685f76"],["E:/GitHubBlog/public/archives/page/4/index.html","d7f044d8ee1588bd29d88d0b29dcd7a7"],["E:/GitHubBlog/public/css/404.css","b1bb50e1fabe41adcec483f6427fb3aa"],["E:/GitHubBlog/public/css/index.css","2389364df340948a5b822422956c2c87"],["E:/GitHubBlog/public/css/var.css","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/img/algolia.svg","fd40b88ac5370a5353a50b8175c1f367"],["E:/GitHubBlog/public/img/avatar.png","6cc4a809d23e3d8946a299ae4ce4e4cd"],["E:/GitHubBlog/public/index.html","9ff0a37355827e94dbcee207caea7d75"],["E:/GitHubBlog/public/js/copy.js","45aae54bf2e43ac27ecc41eb5e0bacf7"],["E:/GitHubBlog/public/js/fancybox.js","f84d626654b9bbc05720952b3effe062"],["E:/GitHubBlog/public/js/fireworks.js","35933ce61c158ef9c33b5e089fe757ac"],["E:/GitHubBlog/public/js/head.js","347edd99f8e3921b45fa617e839d8182"],["E:/GitHubBlog/public/js/hexo-theme-melody.js","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/js/katex.js","d971ba8b8dee1120ef66744b89cfd2b1"],["E:/GitHubBlog/public/js/scroll.js","a12ad7e37b9e325f8da3d0523d3757c7"],["E:/GitHubBlog/public/js/search/algolia.js","53160985d32d6fd66d98aa0e05b081ac"],["E:/GitHubBlog/public/js/search/local-search.js","1565b508bd866ed6fbd724a3858af5d8"],["E:/GitHubBlog/public/js/sidebar.js","d24db780974e661198eeb2e45f20a28f"],["E:/GitHubBlog/public/js/third-party/anime.min.js","9b4bbe6deb700e1c3606eab732f5eea5"],["E:/GitHubBlog/public/js/third-party/canvas-ribbon.js","09c181544ddff1db701db02ac30453e0"],["E:/GitHubBlog/public/js/third-party/jquery.fancybox.min.js","3c9fa1c1199cd4f874d855ecb1641335"],["E:/GitHubBlog/public/js/third-party/jquery.min.js","c9f5aeeca3ad37bf2aa006139b935f0a"],["E:/GitHubBlog/public/js/third-party/reveal/head.min.js","aad121203010122e05f1766d92385214"],["E:/GitHubBlog/public/js/third-party/reveal/highlight.min.js","44594243bec43813a16371af8fe7e105"],["E:/GitHubBlog/public/js/third-party/reveal/markdown.min.js","7ec4cef5a7fe3f0bf0eb4dc6d7bca114"],["E:/GitHubBlog/public/js/third-party/reveal/marked.min.js","c2a88705e206d71dc21fdc4445349127"],["E:/GitHubBlog/public/js/third-party/reveal/math.min.js","0a278fee2e57c530ab07f7d2d9ea8d96"],["E:/GitHubBlog/public/js/third-party/reveal/notes.min.js","89a0dfae4d706f9c75b317f686c3aa14"],["E:/GitHubBlog/public/js/third-party/reveal/reveal.min.js","8988419d67efb5fe93e291a357e26ec9"],["E:/GitHubBlog/public/js/third-party/reveal/zoom.min.js","9791f96e63e7d534cba2b67d4bda0419"],["E:/GitHubBlog/public/js/third-party/velocity.min.js","64da069aba987ea0512cf610600a56d1"],["E:/GitHubBlog/public/js/third-party/velocity.ui.min.js","c8ca438424a080620f7b2f4ee4b0fff1"],["E:/GitHubBlog/public/js/transition.js","911db4268f0f6621073afcced9e1bfef"],["E:/GitHubBlog/public/js/utils.js","3ff3423d966a1c351e9867813b3f6d36"],["E:/GitHubBlog/public/page/2/index.html","def046a4d19613100163bbd8c3e6d1e3"],["E:/GitHubBlog/public/page/3/index.html","ae27993b5686f259d3df227f8cc84866"],["E:/GitHubBlog/public/page/4/index.html","35f5b17345234ab12ed596af437705cb"],["E:/GitHubBlog/public/tags/Android/index.html","dbe732bbfa7ebb68db7d203160a88b30"],["E:/GitHubBlog/public/tags/NLP/index.html","f9b7b18856e925aa9ad69f75fb62d446"],["E:/GitHubBlog/public/tags/NLP/page/2/index.html","9f38ef32fd2248804417ea9a3f2b8252"],["E:/GitHubBlog/public/tags/R/index.html","b1e0fbf0e912ae9abd195b8a41546d2a"],["E:/GitHubBlog/public/tags/index.html","d8a61e9b3d41b18165d898b051f4de95"],["E:/GitHubBlog/public/tags/java/index.html","de6c20381df86e530b9ad8fcb8b0d644"],["E:/GitHubBlog/public/tags/leetcode/index.html","b5580880f45ccea4433089d3641d8cb4"],["E:/GitHubBlog/public/tags/python/index.html","711b6942763496dbd348881dd79ee7bb"],["E:/GitHubBlog/public/tags/总结/index.html","3757e05f9a866f6ce1184100633d09c3"],["E:/GitHubBlog/public/tags/数据分析/index.html","1956c69449baad60187afcf82798b942"],["E:/GitHubBlog/public/tags/数据挖掘/index.html","d215957aad20b8f58158e545d85ca365"],["E:/GitHubBlog/public/tags/数据结构/index.html","03c45064c64913f308587e87cdbec42e"],["E:/GitHubBlog/public/tags/机器学习/index.html","f847adc6e64b8006911ff58219026fb0"],["E:/GitHubBlog/public/tags/深度学习/index.html","73bafac77866d96cb73e922a7249e35f"],["E:/GitHubBlog/public/tags/爬虫/index.html","e7a3f236cf6b29a1a277e95e3f3d5bef"],["E:/GitHubBlog/public/tags/笔记/index.html","a01e22ca9549178e65c9fdf2a26a82bb"],["E:/GitHubBlog/public/tags/论文/index.html","6997f92811b7ce962e00a7e6b34d5738"],["E:/GitHubBlog/public/tags/论文/page/2/index.html","38e04ccc2d3f298184d3be180937d3c8"],["E:/GitHubBlog/public/tags/读书笔记/index.html","85e3db0de633b27259b99de6892dd832"]];
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







