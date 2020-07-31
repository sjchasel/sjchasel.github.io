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

var precacheConfig = [["E:/GitHubBlog/public/2020/01/28/hello-world/index.html","66046a7d17fe73447d6e3f4419e69edc"],["E:/GitHubBlog/public/2020/02/08/python爬虫学习总结/index.html","f26392ef028afc806e6006156b2eff6b"],["E:/GitHubBlog/public/2020/02/19/数据结构第一周-绪论/index.html","e129cdac23b7f5467c1e2bb159d2b4c8"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-1/index.html","336cb96841afe838717b1ca80e3b8907"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-2-创建数据集/index.html","bc9dd726f4511b36b7ba031941ae2996"],["E:/GitHubBlog/public/2020/03/01/《R语言实战》学习总结-3-图形初阶/index.html","d0a41936af7cb7ae222b5940fd814670"],["E:/GitHubBlog/public/2020/03/02/数据结构第二周-线性结构1/index.html","2830b2c0ee0204274abc9e72ad44b8da"],["E:/GitHubBlog/public/2020/03/03/《R语言实战》学习总结-4-基本数据管理/index.html","82fd90cee45c1048f26ad5f47a3de2ab"],["E:/GitHubBlog/public/2020/03/14/数据结构第三周-线性结构2/index.html","1759afa8db91647de66c7b4ac28d01a2"],["E:/GitHubBlog/public/2020/03/14/数据结构第四周-栈与队列/index.html","755e51d628134b22585d003b6bb068db"],["E:/GitHubBlog/public/2020/03/17/leetcode记录-1、88、20/index.html","92be51383e3ebd1a927026b5d5bc9c83"],["E:/GitHubBlog/public/2020/03/20/温度转换程序/index.html","176a910ec9654ba32bf5713f2d88ec29"],["E:/GitHubBlog/public/2020/03/22/数据结构第五周-递归/index.html","cff648ea7aac398e2530d4d06280f056"],["E:/GitHubBlog/public/2020/03/22/数据结构第六周-串与数组/index.html","8c6173384d41dcf17379d200e452e444"],["E:/GitHubBlog/public/2020/04/02/leetcode记录-52/index.html","50a4e59d106f719db82eb7b7060682c0"],["E:/GitHubBlog/public/2020/04/17/Android第八周作业-每天第一次打开应用更新数据/index.html","1b8b73a504bb350dfbd325ff358722a7"],["E:/GitHubBlog/public/2020/05/21/attention模型/index.html","0b2ef1566714281bbd6417152ccef888"],["E:/GitHubBlog/public/2020/05/21/编码器与解码器框架/index.html","576bc3396e801d0aa5a990f228667e93"],["E:/GitHubBlog/public/2020/07/11/思考，快与慢/index.html","3aef882ea2a5f4b47309b490ade9c2b5"],["E:/GitHubBlog/public/2020/07/13/20200713-0717总结/index.html","2a8a8607d3b94543cde5222b089835d9"],["E:/GitHubBlog/public/2020/07/13/ZeroShotCeres Zero-Shot Relation Extraction/index.html","5ad849eb70d3b106f0d10277b20afcd1"],["E:/GitHubBlog/public/2020/07/13/《lingo与excel在数学建模中的应用》/index.html","6d095fa49a7807427484eb99e257afdc"],["E:/GitHubBlog/public/2020/07/20/Keyphrase Generation A Text Summarization Struggle/index.html","322f7923ffdb56d588a0f5f4987606b1"],["E:/GitHubBlog/public/2020/07/20/TPR/index.html","f0b815843dedbf9bef27c801cb33078f"],["E:/GitHubBlog/public/2020/07/21/天池NLP赛事-新闻文本分类-Task1-赛题理解/index.html","ada31d7018e53e6f5938ec8eb6fe878e"],["E:/GitHubBlog/public/2020/07/22/天池NLP赛事-新闻文本分类-Task2-数据读取与数据分析/index.html","cbbe8118ecac5ca1efae821f7249c894"],["E:/GitHubBlog/public/2020/07/23/Keyphrase Generation for Scientific Document Retrieval/index.html","e33a9a6a574813e2881b2e5e3e2b2112"],["E:/GitHubBlog/public/2020/07/24/20200718-0724总结/index.html","542285306891d0bdc85b5aea276af72c"],["E:/GitHubBlog/public/2020/07/25/天池NLP赛事-新闻文本分类-Task3-基于机器学习的文本分类/index.html","469ea5c9d493a45801355cfb1899aa57"],["E:/GitHubBlog/public/2020/07/27/基于深度学习的文本分类/index.html","ece9dcf512f941eabda31eba4d53c73f"],["E:/GitHubBlog/public/2020/07/31/20200725-0731总结/index.html","a84ffe748bf84a40c7abff7139d709df"],["E:/GitHubBlog/public/2020/07/31/Task5/index.html","b8d059a4bb862b9d62c160c45953fc5a"],["E:/GitHubBlog/public/archives/2020/01/index.html","57ce31f6a861380b846e2bcaa314b220"],["E:/GitHubBlog/public/archives/2020/02/index.html","88bee43256bb11110b0b8b4ce6b74f23"],["E:/GitHubBlog/public/archives/2020/03/index.html","2d72e5b8c1fffef01ab792826aa810b2"],["E:/GitHubBlog/public/archives/2020/04/index.html","03fd26c7fb819e73895d14140373a4dd"],["E:/GitHubBlog/public/archives/2020/05/index.html","57653d21948f1d153d6ad07aeffb8f7e"],["E:/GitHubBlog/public/archives/2020/07/index.html","5318c310650120cbf43d0d2cfa3abc2d"],["E:/GitHubBlog/public/archives/2020/07/page/2/index.html","8a6cdc2e2b7d10210b7ac0a94d18c963"],["E:/GitHubBlog/public/archives/2020/index.html","8cf4c2a00b8e6838a4efdbb217dc5d68"],["E:/GitHubBlog/public/archives/2020/page/2/index.html","a5f937d01435a71e1254da3ae704a79f"],["E:/GitHubBlog/public/archives/2020/page/3/index.html","3a3bef5fed9bc18fda115e3b800727f7"],["E:/GitHubBlog/public/archives/2020/page/4/index.html","ecea1a7ca042fdca139a620b4b788dc5"],["E:/GitHubBlog/public/archives/index.html","718b8ec157bd9fb31827214c11e7e461"],["E:/GitHubBlog/public/archives/page/2/index.html","4bfa266fe127fa8d23b9606a9cd1c23e"],["E:/GitHubBlog/public/archives/page/3/index.html","c71f440adecb7be18533ae3fb3f8ae92"],["E:/GitHubBlog/public/archives/page/4/index.html","6422d4741d88bf8efe6ef14f86f71860"],["E:/GitHubBlog/public/css/404.css","b1bb50e1fabe41adcec483f6427fb3aa"],["E:/GitHubBlog/public/css/index.css","2389364df340948a5b822422956c2c87"],["E:/GitHubBlog/public/css/var.css","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/img/algolia.svg","fd40b88ac5370a5353a50b8175c1f367"],["E:/GitHubBlog/public/img/avatar.png","6cc4a809d23e3d8946a299ae4ce4e4cd"],["E:/GitHubBlog/public/index.html","6c78c5d6fb476ee4d8753c6f622abbe8"],["E:/GitHubBlog/public/js/copy.js","45aae54bf2e43ac27ecc41eb5e0bacf7"],["E:/GitHubBlog/public/js/fancybox.js","f84d626654b9bbc05720952b3effe062"],["E:/GitHubBlog/public/js/fireworks.js","35933ce61c158ef9c33b5e089fe757ac"],["E:/GitHubBlog/public/js/head.js","347edd99f8e3921b45fa617e839d8182"],["E:/GitHubBlog/public/js/hexo-theme-melody.js","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/js/katex.js","d971ba8b8dee1120ef66744b89cfd2b1"],["E:/GitHubBlog/public/js/scroll.js","a12ad7e37b9e325f8da3d0523d3757c7"],["E:/GitHubBlog/public/js/search/algolia.js","53160985d32d6fd66d98aa0e05b081ac"],["E:/GitHubBlog/public/js/search/local-search.js","1565b508bd866ed6fbd724a3858af5d8"],["E:/GitHubBlog/public/js/sidebar.js","d24db780974e661198eeb2e45f20a28f"],["E:/GitHubBlog/public/js/third-party/anime.min.js","9b4bbe6deb700e1c3606eab732f5eea5"],["E:/GitHubBlog/public/js/third-party/canvas-ribbon.js","09c181544ddff1db701db02ac30453e0"],["E:/GitHubBlog/public/js/third-party/jquery.fancybox.min.js","3c9fa1c1199cd4f874d855ecb1641335"],["E:/GitHubBlog/public/js/third-party/jquery.min.js","c9f5aeeca3ad37bf2aa006139b935f0a"],["E:/GitHubBlog/public/js/third-party/reveal/head.min.js","aad121203010122e05f1766d92385214"],["E:/GitHubBlog/public/js/third-party/reveal/highlight.min.js","44594243bec43813a16371af8fe7e105"],["E:/GitHubBlog/public/js/third-party/reveal/markdown.min.js","7ec4cef5a7fe3f0bf0eb4dc6d7bca114"],["E:/GitHubBlog/public/js/third-party/reveal/marked.min.js","c2a88705e206d71dc21fdc4445349127"],["E:/GitHubBlog/public/js/third-party/reveal/math.min.js","0a278fee2e57c530ab07f7d2d9ea8d96"],["E:/GitHubBlog/public/js/third-party/reveal/notes.min.js","89a0dfae4d706f9c75b317f686c3aa14"],["E:/GitHubBlog/public/js/third-party/reveal/reveal.min.js","8988419d67efb5fe93e291a357e26ec9"],["E:/GitHubBlog/public/js/third-party/reveal/zoom.min.js","9791f96e63e7d534cba2b67d4bda0419"],["E:/GitHubBlog/public/js/third-party/velocity.min.js","64da069aba987ea0512cf610600a56d1"],["E:/GitHubBlog/public/js/third-party/velocity.ui.min.js","c8ca438424a080620f7b2f4ee4b0fff1"],["E:/GitHubBlog/public/js/transition.js","911db4268f0f6621073afcced9e1bfef"],["E:/GitHubBlog/public/js/utils.js","3ff3423d966a1c351e9867813b3f6d36"],["E:/GitHubBlog/public/page/2/index.html","9721be20fc5a7c269706722fe8b27a37"],["E:/GitHubBlog/public/page/3/index.html","12fd4753374dacc561ddfff6814d1c1e"],["E:/GitHubBlog/public/page/4/index.html","f73f71a7d2a17846adcbd135f1c58921"],["E:/GitHubBlog/public/tags/Android/index.html","fdf17bddb38c4ef5adb8bbd104f64642"],["E:/GitHubBlog/public/tags/NLP/index.html","0ad2b6e6bd0288bc8f36f0c706b841e4"],["E:/GitHubBlog/public/tags/NLP/page/2/index.html","743219345f1fac92105ab7fec9abce06"],["E:/GitHubBlog/public/tags/R/index.html","09376228ddc8a6be3bf6ddbcb49088e2"],["E:/GitHubBlog/public/tags/index.html","89aa64098b7c346539ecddae174ffc66"],["E:/GitHubBlog/public/tags/java/index.html","e09997b5a89daedeaa9a98d39263c4e2"],["E:/GitHubBlog/public/tags/leetcode/index.html","2d7ac9138a94cbb7dc7bcaef15701030"],["E:/GitHubBlog/public/tags/lingo/index.html","0e9b12de0070e18a8c3f865625d6362a"],["E:/GitHubBlog/public/tags/python/index.html","331026fe57a0bb641b70879e3f66f36d"],["E:/GitHubBlog/public/tags/总结/index.html","044daad5e66485abde746af2c96ebc0a"],["E:/GitHubBlog/public/tags/数据分析/index.html","9da87914d255a7bd06533dfe53c5c0a3"],["E:/GitHubBlog/public/tags/数据结构/index.html","0e4658d1a8102da074e33256736ba984"],["E:/GitHubBlog/public/tags/数模/index.html","22fcf4de83c024b9cb35ce174e105d3a"],["E:/GitHubBlog/public/tags/比赛/index.html","de9831c3e20078974b8d8e6e97dd5e18"],["E:/GitHubBlog/public/tags/深度学习/index.html","8783e0ac0577f9f12c0e00d6bdff093e"],["E:/GitHubBlog/public/tags/爬虫/index.html","a7eb92d63cca20f01c6a4d73388e8c29"],["E:/GitHubBlog/public/tags/论文/index.html","99f8138f06b759d94eec2705e8445f1c"],["E:/GitHubBlog/public/tags/读书笔记/index.html","d58d99beb5e177ffefff905ae850d31c"]];
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







