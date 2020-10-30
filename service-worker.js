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

var precacheConfig = [["E:/GitHubBlog/public/2020/01/28/hello-world/index.html","a1153868a6a599bf3c8f3f49550e600b"],["E:/GitHubBlog/public/2020/02/08/python爬虫学习总结/index.html","11ca437025e586c72d7f208d537e931b"],["E:/GitHubBlog/public/2020/02/19/数据结构第一周-绪论/index.html","a8fe4f64bb806405dd27c442e2a7eb10"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-1/index.html","f0dfc8cfecdb9ef186478b9eefe3d62b"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-2-创建数据集/index.html","3e3722e5adc08d2447779f42d8235802"],["E:/GitHubBlog/public/2020/03/01/《R语言实战》学习总结-3-图形初阶/index.html","da48f3ec16a303545f51d0f45eb79d18"],["E:/GitHubBlog/public/2020/03/02/数据结构第二周-线性结构1/index.html","b9b240651bcde58b8c09b88b8935532e"],["E:/GitHubBlog/public/2020/03/03/《R语言实战》学习总结-4-基本数据管理/index.html","d1d722c1dc161467ffdb154154eaa637"],["E:/GitHubBlog/public/2020/03/14/数据结构第三周-线性结构2/index.html","fcbaa068c396f2d74731ee20a3eb59ee"],["E:/GitHubBlog/public/2020/03/14/数据结构第四周-栈与队列/index.html","8255537f1b294e20fcad1b0ae8513750"],["E:/GitHubBlog/public/2020/03/17/leetcode记录-1、88、20/index.html","b620b46d5078ac9dfa37a6103df045bc"],["E:/GitHubBlog/public/2020/03/20/温度转换程序/index.html","c8d9addf14247722e50443b03c65f157"],["E:/GitHubBlog/public/2020/03/22/数据结构第五周-递归/index.html","2b0eb5b2d0fa4ebb99f3f02071bbd4f8"],["E:/GitHubBlog/public/2020/03/22/数据结构第六周-串与数组/index.html","f8b9ef08b8eb3672fdeb6410c6492c94"],["E:/GitHubBlog/public/2020/04/02/leetcode记录-52/index.html","cdf2efbae6b01ccb211511a78018f180"],["E:/GitHubBlog/public/2020/04/17/Android第八周作业-每天第一次打开应用更新数据/index.html","25619606d30323c87658f9eb0669cf3a"],["E:/GitHubBlog/public/2020/05/21/attention模型/index.html","3056fdbc7778b3d13d355a5faa9fe583"],["E:/GitHubBlog/public/2020/05/21/编码器与解码器框架/index.html","8d62dd31ed18e1c19d819ee7380fd8bc"],["E:/GitHubBlog/public/2020/07/11/思考，快与慢/index.html","094cd9fd2778027cbb3d09a70616d433"],["E:/GitHubBlog/public/2020/07/13/20200713-0717总结/index.html","169f487a233b63bb40ec3b35bbbc9000"],["E:/GitHubBlog/public/2020/07/13/ZeroShotCeres Zero-Shot Relation Extraction/index.html","c1c79a4b6304da7124f378c3fd023955"],["E:/GitHubBlog/public/2020/07/20/Keyphrase Generation A Text Summarization Struggle/index.html","073409dcfd2ba86e99db585aef679afe"],["E:/GitHubBlog/public/2020/07/20/TPR/index.html","57b68be50a4c6a5d0c7316fff76f54dc"],["E:/GitHubBlog/public/2020/07/23/Keyphrase Generation for Scientific Document Retrieval/index.html","0122340334538c99fb3f99a6dfd37582"],["E:/GitHubBlog/public/2020/07/24/20200718-0724总结/index.html","b0198af7a4fd2b270ebb2a053fb3a79b"],["E:/GitHubBlog/public/2020/07/31/20200725-0731总结/index.html","a3919bef3c8f6f3677cb1651cb4dcba7"],["E:/GitHubBlog/public/2020/08/07/20200801-0807总结/index.html","9f74f3a2be9970620bdce0f3a066f19a"],["E:/GitHubBlog/public/2020/08/12/逻辑回归算法/index.html","9af3413a0b48bfe4205480eff7ff319b"],["E:/GitHubBlog/public/2020/08/13/李宏毅-机器学习21/index.html","6ff97814ff70b33c8c793736c1a4eaa2"],["E:/GitHubBlog/public/2020/08/14/20200808-0815总结/index.html","5b2a8df3ea701a3fb3aed248b98f1917"],["E:/GitHubBlog/public/2020/09/03/20200816-0903总结/index.html","f3e950d14fb6b198295de5195614871f"],["E:/GitHubBlog/public/2020/09/16/20200914-0918总结/index.html","82c3e8fee18ba730f307be603b20cf22"],["E:/GitHubBlog/public/2020/09/25/20200919-0925总结/index.html","044de6a5fbeb3ac6fb0b80568a3ee161"],["E:/GitHubBlog/public/2020/10/06/《Deep Keyphrase Generation》详细解析！/index.html","e4bdc8cd369e2cb53bb022ffde3b579a"],["E:/GitHubBlog/public/2020/10/16/20201008-1017总结/index.html","2b6a8445684bfd28122d5ac586096c4b"],["E:/GitHubBlog/public/2020/10/20/特征选择的几种方法(一)————Filter过滤法/index.html","4304915dcc8a5ae9e503314c2ec35557"],["E:/GitHubBlog/public/2020/10/21/PCA/index.html","cebac2a8bb0482227efbdab10c2714b4"],["E:/GitHubBlog/public/2020/10/22/20201018-1023总结/index.html","4e20961bec98f4432a1a0a3af49b2acb"],["E:/GitHubBlog/public/2020/10/24/20201024-1030/index.html","57ec32bfefe7375ee2fcaeacd793c31a"],["E:/GitHubBlog/public/archives/2020/01/index.html","7cd6858999625b450394b4d2ae4059d6"],["E:/GitHubBlog/public/archives/2020/02/index.html","02250160312310b54d6b7ad04e040c47"],["E:/GitHubBlog/public/archives/2020/03/index.html","8a6100c6cb7834adedb306018aed0afd"],["E:/GitHubBlog/public/archives/2020/04/index.html","6fbfacc588342810e01c5eb3e1bd7cd0"],["E:/GitHubBlog/public/archives/2020/05/index.html","8ccbdbe50261c1b466631f55e7889f79"],["E:/GitHubBlog/public/archives/2020/07/index.html","bc352c77b659fae871c34495debdfd11"],["E:/GitHubBlog/public/archives/2020/08/index.html","9394e761cdcd11abed93ec521c7b291c"],["E:/GitHubBlog/public/archives/2020/09/index.html","f3cc344517a621740b1e75bc8810ecfb"],["E:/GitHubBlog/public/archives/2020/10/index.html","005a02f7e8f8264888b627fe7709498f"],["E:/GitHubBlog/public/archives/2020/index.html","a5ec5e4beaa3c8771de592a9f5e8167b"],["E:/GitHubBlog/public/archives/2020/page/2/index.html","a3c0d4e9ab2ac6ec090fd21d492e9b11"],["E:/GitHubBlog/public/archives/2020/page/3/index.html","49fd1f9900731e0c9cf0de050e668a57"],["E:/GitHubBlog/public/archives/2020/page/4/index.html","f81f5897fb70618c305daf560602b145"],["E:/GitHubBlog/public/archives/index.html","4e3fc616e83deacd72541a8274ce8c18"],["E:/GitHubBlog/public/archives/page/2/index.html","01a013e7884820ec55b1af329defcb35"],["E:/GitHubBlog/public/archives/page/3/index.html","6da455f52919e69b45b6039eafc73fa5"],["E:/GitHubBlog/public/archives/page/4/index.html","afc95f0ee18ae06448988b9948049053"],["E:/GitHubBlog/public/css/404.css","b1bb50e1fabe41adcec483f6427fb3aa"],["E:/GitHubBlog/public/css/index.css","2389364df340948a5b822422956c2c87"],["E:/GitHubBlog/public/css/var.css","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/img/algolia.svg","fd40b88ac5370a5353a50b8175c1f367"],["E:/GitHubBlog/public/img/avatar.png","6cc4a809d23e3d8946a299ae4ce4e4cd"],["E:/GitHubBlog/public/index.html","7dd8160c01d0023c9af134d79d7bedbe"],["E:/GitHubBlog/public/js/copy.js","45aae54bf2e43ac27ecc41eb5e0bacf7"],["E:/GitHubBlog/public/js/fancybox.js","f84d626654b9bbc05720952b3effe062"],["E:/GitHubBlog/public/js/fireworks.js","35933ce61c158ef9c33b5e089fe757ac"],["E:/GitHubBlog/public/js/head.js","347edd99f8e3921b45fa617e839d8182"],["E:/GitHubBlog/public/js/hexo-theme-melody.js","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/js/katex.js","d971ba8b8dee1120ef66744b89cfd2b1"],["E:/GitHubBlog/public/js/scroll.js","a12ad7e37b9e325f8da3d0523d3757c7"],["E:/GitHubBlog/public/js/search/algolia.js","53160985d32d6fd66d98aa0e05b081ac"],["E:/GitHubBlog/public/js/search/local-search.js","1565b508bd866ed6fbd724a3858af5d8"],["E:/GitHubBlog/public/js/sidebar.js","d24db780974e661198eeb2e45f20a28f"],["E:/GitHubBlog/public/js/third-party/anime.min.js","9b4bbe6deb700e1c3606eab732f5eea5"],["E:/GitHubBlog/public/js/third-party/canvas-ribbon.js","09c181544ddff1db701db02ac30453e0"],["E:/GitHubBlog/public/js/third-party/jquery.fancybox.min.js","3c9fa1c1199cd4f874d855ecb1641335"],["E:/GitHubBlog/public/js/third-party/jquery.min.js","c9f5aeeca3ad37bf2aa006139b935f0a"],["E:/GitHubBlog/public/js/third-party/reveal/head.min.js","aad121203010122e05f1766d92385214"],["E:/GitHubBlog/public/js/third-party/reveal/highlight.min.js","44594243bec43813a16371af8fe7e105"],["E:/GitHubBlog/public/js/third-party/reveal/markdown.min.js","7ec4cef5a7fe3f0bf0eb4dc6d7bca114"],["E:/GitHubBlog/public/js/third-party/reveal/marked.min.js","c2a88705e206d71dc21fdc4445349127"],["E:/GitHubBlog/public/js/third-party/reveal/math.min.js","0a278fee2e57c530ab07f7d2d9ea8d96"],["E:/GitHubBlog/public/js/third-party/reveal/notes.min.js","89a0dfae4d706f9c75b317f686c3aa14"],["E:/GitHubBlog/public/js/third-party/reveal/reveal.min.js","8988419d67efb5fe93e291a357e26ec9"],["E:/GitHubBlog/public/js/third-party/reveal/zoom.min.js","9791f96e63e7d534cba2b67d4bda0419"],["E:/GitHubBlog/public/js/third-party/velocity.min.js","64da069aba987ea0512cf610600a56d1"],["E:/GitHubBlog/public/js/third-party/velocity.ui.min.js","c8ca438424a080620f7b2f4ee4b0fff1"],["E:/GitHubBlog/public/js/transition.js","911db4268f0f6621073afcced9e1bfef"],["E:/GitHubBlog/public/js/utils.js","3ff3423d966a1c351e9867813b3f6d36"],["E:/GitHubBlog/public/page/2/index.html","f572e6e72523405986465623880dcabe"],["E:/GitHubBlog/public/page/3/index.html","288bb117c024f5e5cd5fabd69f623560"],["E:/GitHubBlog/public/page/4/index.html","47b3510f858d1a7a5c612f36b7326dc6"],["E:/GitHubBlog/public/tags/Android/index.html","31db3081b6fad6b9c5dd9113eafb186d"],["E:/GitHubBlog/public/tags/NLP/index.html","35c382777e1bf1eff2c24675420cc9e9"],["E:/GitHubBlog/public/tags/NLP/page/2/index.html","d7f09d23bd90a771cb51154bf073a157"],["E:/GitHubBlog/public/tags/R/index.html","35fd3bf1adc1cb02dfb7aff4993a7270"],["E:/GitHubBlog/public/tags/index.html","8357f486e74ed0ea73bee66a4d66435d"],["E:/GitHubBlog/public/tags/java/index.html","37cf105c70e92e671e589d3d13e758f8"],["E:/GitHubBlog/public/tags/leetcode/index.html","27263311ac5594a97cd2bb52c3a3214b"],["E:/GitHubBlog/public/tags/python/index.html","bc4c7b176b66686c56fb72059da6c3a2"],["E:/GitHubBlog/public/tags/总结/index.html","97ec1a6a7269fb457794edb8cf6b6378"],["E:/GitHubBlog/public/tags/总结/page/2/index.html","a2f17f52f04b5d3e8aaf0a119361c1a7"],["E:/GitHubBlog/public/tags/数据分析/index.html","e64409ace24821ff479429a6cf56fbf9"],["E:/GitHubBlog/public/tags/数据挖掘/index.html","f5dad6597d4096ecdf9b44f6d37dbde0"],["E:/GitHubBlog/public/tags/数据结构/index.html","96e7d5d3f3fb29f979097677cf074f76"],["E:/GitHubBlog/public/tags/机器学习/index.html","115cd5835c50dc3aea1728a567070592"],["E:/GitHubBlog/public/tags/深度学习/index.html","f7f4f141abf5f9f189fca4aa7d041b40"],["E:/GitHubBlog/public/tags/爬虫/index.html","24a7d73d0189e8a68a7fe610ffddc360"],["E:/GitHubBlog/public/tags/笔记/index.html","806cbcb32149e93c7db6cbba243d42e6"],["E:/GitHubBlog/public/tags/论文/index.html","7f8060eead24f7fa688cb2bb785d6348"],["E:/GitHubBlog/public/tags/论文/page/2/index.html","d75c66608ffb053e385eef3e555e4f65"],["E:/GitHubBlog/public/tags/读书笔记/index.html","99318f75c15f3482e8e087ab02ae0f83"]];
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







