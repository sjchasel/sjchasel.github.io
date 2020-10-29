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

var precacheConfig = [["E:/GitHubBlog/public/2020/01/28/hello-world/index.html","025ba81d407533654b004e3b5a16523e"],["E:/GitHubBlog/public/2020/02/08/python爬虫学习总结/index.html","a75b5d73985bc2f413ae83a05946e3da"],["E:/GitHubBlog/public/2020/02/19/数据结构第一周-绪论/index.html","6e072cbefcac23a6efdbaf008327c6d3"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-1/index.html","5fd0da34f960c2cc9de127e476bc9ccd"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-2-创建数据集/index.html","c31aa7e65dee87e3a8d042a4e4f36764"],["E:/GitHubBlog/public/2020/03/01/《R语言实战》学习总结-3-图形初阶/index.html","6f7d4372ca8de349c33fbb4a48e7b609"],["E:/GitHubBlog/public/2020/03/02/数据结构第二周-线性结构1/index.html","9f94cb17905564f1aa5a05462d565003"],["E:/GitHubBlog/public/2020/03/03/《R语言实战》学习总结-4-基本数据管理/index.html","40e0cc411db6d73890e82d646cac3b45"],["E:/GitHubBlog/public/2020/03/14/数据结构第三周-线性结构2/index.html","499050faa114fcd84d8d35d2fd596d68"],["E:/GitHubBlog/public/2020/03/14/数据结构第四周-栈与队列/index.html","90cb23cc6dfbc1cae5420da1a5600d5a"],["E:/GitHubBlog/public/2020/03/17/leetcode记录-1、88、20/index.html","084a748f483781173508cf5dbc102544"],["E:/GitHubBlog/public/2020/03/20/温度转换程序/index.html","248169e892b28b30efe2fa26035d1d93"],["E:/GitHubBlog/public/2020/03/22/数据结构第五周-递归/index.html","7dbfc927b94d544725d06142fc51cc06"],["E:/GitHubBlog/public/2020/03/22/数据结构第六周-串与数组/index.html","c8fc3d6f91843312f6df78bebe6cb6b3"],["E:/GitHubBlog/public/2020/04/02/leetcode记录-52/index.html","1b4eee0cc65ded6277aabd426bb860a3"],["E:/GitHubBlog/public/2020/04/17/Android第八周作业-每天第一次打开应用更新数据/index.html","10840977ae20efca9bad9a59bdb4900f"],["E:/GitHubBlog/public/2020/05/21/attention模型/index.html","a4a40735733401bacf1363843fc61e73"],["E:/GitHubBlog/public/2020/05/21/编码器与解码器框架/index.html","5fc093f9d2b0b58506ba104c441c13f6"],["E:/GitHubBlog/public/2020/07/11/思考，快与慢/index.html","cb48ce14e4685deaf53e09a65a53ffef"],["E:/GitHubBlog/public/2020/07/13/20200713-0717总结/index.html","81b8053cf6275a72e7417ca9661f1225"],["E:/GitHubBlog/public/2020/07/13/ZeroShotCeres Zero-Shot Relation Extraction/index.html","cd5b77e577b665d1fdb2b75a827d97ff"],["E:/GitHubBlog/public/2020/07/20/Keyphrase Generation A Text Summarization Struggle/index.html","ef790e378aff02f67d9f799c96ebdac0"],["E:/GitHubBlog/public/2020/07/20/TPR/index.html","52d6f7df3165e8ae4b1f48d1d1eaae2f"],["E:/GitHubBlog/public/2020/07/23/Keyphrase Generation for Scientific Document Retrieval/index.html","bef3c04f3c8391abcc86a0cb031f4093"],["E:/GitHubBlog/public/2020/07/24/20200718-0724总结/index.html","f9a797d592d69663023c86f71e4a420c"],["E:/GitHubBlog/public/2020/07/31/20200725-0731总结/index.html","9c466f773ddafdaf541b8d72f65ed700"],["E:/GitHubBlog/public/2020/08/07/20200801-0807总结/index.html","78e277f8175b509b7f4b8a938952722b"],["E:/GitHubBlog/public/2020/08/12/逻辑回归算法/index.html","0e81b62e0ba4f2132d5ebe3c333b851f"],["E:/GitHubBlog/public/2020/08/13/李宏毅-机器学习21/index.html","a3d5c7741741e78699bc85381dea1c70"],["E:/GitHubBlog/public/2020/08/14/20200808-0815总结/index.html","6d0495e79b6950cbc9419c84438cd09e"],["E:/GitHubBlog/public/2020/09/03/20200816-0903总结/index.html","37b3a4d9082dc292e91ae31e125681d2"],["E:/GitHubBlog/public/2020/09/16/20200914-0918总结/index.html","9566af154349c27b8ab35c9a3768b833"],["E:/GitHubBlog/public/2020/09/25/20200919-0925总结/index.html","41297fe1d3d4798376f56367076ca4a0"],["E:/GitHubBlog/public/2020/10/06/《Deep Keyphrase Generation》详细解析！/index.html","1aa2b666813932c98ed7126dda7d0155"],["E:/GitHubBlog/public/2020/10/16/20201008-1017总结/index.html","fbdfda6e40282175b035a0dbd48af3a5"],["E:/GitHubBlog/public/2020/10/20/特征选择的几种方法(一)————Filter过滤法/index.html","8884ac0abf88be3f8d1a83bf8d491352"],["E:/GitHubBlog/public/2020/10/21/PCA/index.html","d973e0a8b0ab31dcd084497b969eca39"],["E:/GitHubBlog/public/2020/10/22/20201018-1023总结/index.html","631a439a6d4bdf945a0877a87a3bb1cb"],["E:/GitHubBlog/public/2020/10/24/20201024-1030/index.html","fddc38ed95b6b88066c5fc8c7e053b68"],["E:/GitHubBlog/public/2020/10/24/新网银行统计建模大赛记录/index.html","933b25873a3bee9ddcfd12de22c170d1"],["E:/GitHubBlog/public/archives/2020/01/index.html","e24c7b5db20e4ec6c14b74b2a84c251b"],["E:/GitHubBlog/public/archives/2020/02/index.html","1e26b34dfb488a0444039afa2ea5e611"],["E:/GitHubBlog/public/archives/2020/03/index.html","6d6b0a555716105c8d66f17b15a0d078"],["E:/GitHubBlog/public/archives/2020/04/index.html","db40a474625096a557811a3643811eed"],["E:/GitHubBlog/public/archives/2020/05/index.html","3218c60f3fb4ff5c2447ffcd65bdbaaa"],["E:/GitHubBlog/public/archives/2020/07/index.html","a98f9fc80539d4c30056ea81cd33686f"],["E:/GitHubBlog/public/archives/2020/08/index.html","af837b3293d73c0e19d14b28a17b68d5"],["E:/GitHubBlog/public/archives/2020/09/index.html","a277001e8070c7fa2e3abad13ac1a3e0"],["E:/GitHubBlog/public/archives/2020/10/index.html","b4740693666cdf229e8034cce94f8e9a"],["E:/GitHubBlog/public/archives/2020/index.html","4211f99ae716f95034fb335fef41211b"],["E:/GitHubBlog/public/archives/2020/page/2/index.html","d48997aaa6658cecca5274f5ff7def2f"],["E:/GitHubBlog/public/archives/2020/page/3/index.html","d08999b4f9bf9b64f48702f264d8d209"],["E:/GitHubBlog/public/archives/2020/page/4/index.html","bc9efc0ad5b6513907d53a01be531347"],["E:/GitHubBlog/public/archives/index.html","58560b4a8edc5a779bd0afaa023fcc2d"],["E:/GitHubBlog/public/archives/page/2/index.html","6dbe046b97ed899305550c267e1f8837"],["E:/GitHubBlog/public/archives/page/3/index.html","7aa606be80745feee16740dbc5fee302"],["E:/GitHubBlog/public/archives/page/4/index.html","1eeb6eb73761d5d9cf03b2cb84b36b99"],["E:/GitHubBlog/public/css/404.css","b1bb50e1fabe41adcec483f6427fb3aa"],["E:/GitHubBlog/public/css/index.css","2389364df340948a5b822422956c2c87"],["E:/GitHubBlog/public/css/var.css","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/img/algolia.svg","fd40b88ac5370a5353a50b8175c1f367"],["E:/GitHubBlog/public/img/avatar.png","6cc4a809d23e3d8946a299ae4ce4e4cd"],["E:/GitHubBlog/public/index.html","436cedb87b4fea9ac4de3754ef38977c"],["E:/GitHubBlog/public/js/copy.js","45aae54bf2e43ac27ecc41eb5e0bacf7"],["E:/GitHubBlog/public/js/fancybox.js","f84d626654b9bbc05720952b3effe062"],["E:/GitHubBlog/public/js/fireworks.js","35933ce61c158ef9c33b5e089fe757ac"],["E:/GitHubBlog/public/js/head.js","347edd99f8e3921b45fa617e839d8182"],["E:/GitHubBlog/public/js/hexo-theme-melody.js","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/js/katex.js","d971ba8b8dee1120ef66744b89cfd2b1"],["E:/GitHubBlog/public/js/scroll.js","a12ad7e37b9e325f8da3d0523d3757c7"],["E:/GitHubBlog/public/js/search/algolia.js","53160985d32d6fd66d98aa0e05b081ac"],["E:/GitHubBlog/public/js/search/local-search.js","1565b508bd866ed6fbd724a3858af5d8"],["E:/GitHubBlog/public/js/sidebar.js","d24db780974e661198eeb2e45f20a28f"],["E:/GitHubBlog/public/js/third-party/anime.min.js","9b4bbe6deb700e1c3606eab732f5eea5"],["E:/GitHubBlog/public/js/third-party/canvas-ribbon.js","09c181544ddff1db701db02ac30453e0"],["E:/GitHubBlog/public/js/third-party/jquery.fancybox.min.js","3c9fa1c1199cd4f874d855ecb1641335"],["E:/GitHubBlog/public/js/third-party/jquery.min.js","c9f5aeeca3ad37bf2aa006139b935f0a"],["E:/GitHubBlog/public/js/third-party/reveal/head.min.js","aad121203010122e05f1766d92385214"],["E:/GitHubBlog/public/js/third-party/reveal/highlight.min.js","44594243bec43813a16371af8fe7e105"],["E:/GitHubBlog/public/js/third-party/reveal/markdown.min.js","7ec4cef5a7fe3f0bf0eb4dc6d7bca114"],["E:/GitHubBlog/public/js/third-party/reveal/marked.min.js","c2a88705e206d71dc21fdc4445349127"],["E:/GitHubBlog/public/js/third-party/reveal/math.min.js","0a278fee2e57c530ab07f7d2d9ea8d96"],["E:/GitHubBlog/public/js/third-party/reveal/notes.min.js","89a0dfae4d706f9c75b317f686c3aa14"],["E:/GitHubBlog/public/js/third-party/reveal/reveal.min.js","8988419d67efb5fe93e291a357e26ec9"],["E:/GitHubBlog/public/js/third-party/reveal/zoom.min.js","9791f96e63e7d534cba2b67d4bda0419"],["E:/GitHubBlog/public/js/third-party/velocity.min.js","64da069aba987ea0512cf610600a56d1"],["E:/GitHubBlog/public/js/third-party/velocity.ui.min.js","c8ca438424a080620f7b2f4ee4b0fff1"],["E:/GitHubBlog/public/js/transition.js","911db4268f0f6621073afcced9e1bfef"],["E:/GitHubBlog/public/js/utils.js","3ff3423d966a1c351e9867813b3f6d36"],["E:/GitHubBlog/public/page/2/index.html","c0de6e31909b0c4315053499c9c72999"],["E:/GitHubBlog/public/page/3/index.html","87cdcf5a73497abccdf8b91bab0e26d9"],["E:/GitHubBlog/public/page/4/index.html","e937de170906930b652e09dca0a2c2b8"],["E:/GitHubBlog/public/tags/Android/index.html","20a4462ab527c4cb8d696c2755527419"],["E:/GitHubBlog/public/tags/NLP/index.html","d64fdc255557b7831be256b4abc9c8fc"],["E:/GitHubBlog/public/tags/NLP/page/2/index.html","a7097676d87c010560da0b563e5f00a2"],["E:/GitHubBlog/public/tags/R/index.html","31299c4b0afe500928b5d61d926335c0"],["E:/GitHubBlog/public/tags/index.html","d530cbb0ec358bf807cd280a78e63735"],["E:/GitHubBlog/public/tags/java/index.html","07488907e6e6c27df9373a75486e36d5"],["E:/GitHubBlog/public/tags/leetcode/index.html","bbb112a454ebff6aaf2a57a9ac364440"],["E:/GitHubBlog/public/tags/python/index.html","7b8347fd1f224ef5e8b493ae72651607"],["E:/GitHubBlog/public/tags/总结/index.html","6f19bee366ac4644e3691cebec595ff5"],["E:/GitHubBlog/public/tags/总结/page/2/index.html","17893fc31a82c521c1c462597f7c00d6"],["E:/GitHubBlog/public/tags/数据分析/index.html","796f93a018566f58a1abed5b87ca69b2"],["E:/GitHubBlog/public/tags/数据挖掘/index.html","370c46293673e9b102b74bf326b7a004"],["E:/GitHubBlog/public/tags/数据科学/index.html","2a79bc3c73ac869711a30a667adadd84"],["E:/GitHubBlog/public/tags/数据结构/index.html","92b95b4dca18ca9772862c1c388956d6"],["E:/GitHubBlog/public/tags/机器学习/index.html","c1584f8f44b9c96699239e5eeed47030"],["E:/GitHubBlog/public/tags/深度学习/index.html","7a76a33eebc640c68cde6ce5efaeea91"],["E:/GitHubBlog/public/tags/爬虫/index.html","16d611ef7a4cf79489e9bec5d270abb7"],["E:/GitHubBlog/public/tags/竞赛/index.html","3e40df2c64e4e9786e18a35aaea3f8ab"],["E:/GitHubBlog/public/tags/笔记/index.html","5fd46ef500586d1bc9a51332c317143c"],["E:/GitHubBlog/public/tags/论文/index.html","dad79e60e72193262de1ac3dde43a89e"],["E:/GitHubBlog/public/tags/论文/page/2/index.html","b7853cdf2996466220958c545a63336a"],["E:/GitHubBlog/public/tags/读书笔记/index.html","ba2635c953cbb1dd2c2b44b4d7954bb3"]];
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







