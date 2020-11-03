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

var precacheConfig = [["E:/GitHubBlog/public/2020/01/28/hello-world/index.html","b54b660052c23db6f73d1ae5af1f1bcc"],["E:/GitHubBlog/public/2020/02/08/python爬虫学习总结/index.html","7042ab71c9758f02e047ac08848ea6d5"],["E:/GitHubBlog/public/2020/02/19/数据结构第一周-绪论/index.html","ea2c7aabb6e379a8a945e44906c2b61d"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-1/index.html","5bfc20d58732b5b758e7c29237c6780d"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-2-创建数据集/index.html","9dbf13d17ba716224d419ac5a11be0b1"],["E:/GitHubBlog/public/2020/03/01/《R语言实战》学习总结-3-图形初阶/index.html","5d02afe4cde1ce7f42ee0755a5457b2b"],["E:/GitHubBlog/public/2020/03/02/数据结构第二周-线性结构1/index.html","7c555c9504b9f63c62265d27374c563f"],["E:/GitHubBlog/public/2020/03/03/《R语言实战》学习总结-4-基本数据管理/index.html","c91b19180e7381b73b8a5cc551586f61"],["E:/GitHubBlog/public/2020/03/14/数据结构第三周-线性结构2/index.html","0247a8196d1246f9c908036f73b9bcfc"],["E:/GitHubBlog/public/2020/03/14/数据结构第四周-栈与队列/index.html","6b8b95c1f81829e629cd032736d0ceb8"],["E:/GitHubBlog/public/2020/03/17/leetcode记录-1、88、20/index.html","caf8b9207cbcbd7ae07c4e58a6ed2e2e"],["E:/GitHubBlog/public/2020/03/20/温度转换程序/index.html","606816699b0f22c3ca8eb32af3256814"],["E:/GitHubBlog/public/2020/03/22/数据结构第五周-递归/index.html","1b7fdcc47a442983740203a6233b0adb"],["E:/GitHubBlog/public/2020/03/22/数据结构第六周-串与数组/index.html","e15a6ce6b773d70fdf500dec2ff4af73"],["E:/GitHubBlog/public/2020/04/02/leetcode记录-52/index.html","2f6117ff059dd6424e80eb5107db476f"],["E:/GitHubBlog/public/2020/04/17/Android第八周作业-每天第一次打开应用更新数据/index.html","c03079595fd619c7434a6aa549d8827c"],["E:/GitHubBlog/public/2020/05/21/attention模型/index.html","44ee1ce129e5e400207a6be27f2797e3"],["E:/GitHubBlog/public/2020/05/21/编码器与解码器框架/index.html","25455a8caa922564d70e7921836fcef7"],["E:/GitHubBlog/public/2020/07/11/思考，快与慢/index.html","3bf81f49707dd147b36f8f3578b0d8c2"],["E:/GitHubBlog/public/2020/07/13/20200713-0717总结/index.html","56c601ccf73969b30bfbad6e4d11b234"],["E:/GitHubBlog/public/2020/07/13/ZeroShotCeres Zero-Shot Relation Extraction/index.html","534527cb734bd25aa6d79df20f785202"],["E:/GitHubBlog/public/2020/07/20/Keyphrase Generation A Text Summarization Struggle/index.html","e8e3e89f6dd669f2855d1664501d027e"],["E:/GitHubBlog/public/2020/07/20/TPR/index.html","34a8a53328208a507b5d3b263da0083b"],["E:/GitHubBlog/public/2020/07/23/Keyphrase Generation for Scientific Document Retrieval/index.html","03c7e2d8e25f6e02526f1d71d7273855"],["E:/GitHubBlog/public/2020/07/24/20200718-0724总结/index.html","401a387852a3156e1e55c499dfdf859e"],["E:/GitHubBlog/public/2020/07/31/20200725-0731总结/index.html","2ad2f37f6bcfcf5996ac9ec624fd3458"],["E:/GitHubBlog/public/2020/08/07/20200801-0807总结/index.html","6e9a23cdf08737c8c4f3537ca6833092"],["E:/GitHubBlog/public/2020/08/12/逻辑回归算法/index.html","c5fd6706e6ab19962a23ce514ddf1316"],["E:/GitHubBlog/public/2020/08/13/李宏毅-机器学习21/index.html","c63960dbd0190a223ac926307818a503"],["E:/GitHubBlog/public/2020/08/14/20200808-0815总结/index.html","26d9ad3d70f4001cf9738cd23302dd6b"],["E:/GitHubBlog/public/2020/09/03/20200816-0903总结/index.html","38f68f221123aa9234d3ca03f2f2caa6"],["E:/GitHubBlog/public/2020/09/16/20200914-0918总结/index.html","8b8b5643ecb44ef1b97304b5a12b9e00"],["E:/GitHubBlog/public/2020/09/25/20200919-0925总结/index.html","fd2cbc4a250e889a086010f179f1eff3"],["E:/GitHubBlog/public/2020/10/06/《Deep Keyphrase Generation》详细解析！/index.html","0fa9fbf357e998fe397585ac9a25ca55"],["E:/GitHubBlog/public/2020/10/16/20201008-1017总结/index.html","b5831aa8268a302feb970f33c035a60a"],["E:/GitHubBlog/public/2020/10/20/特征选择的几种方法(一)————Filter过滤法/index.html","2327c02b930e56fb07a1860dced0101e"],["E:/GitHubBlog/public/2020/10/21/PCA/index.html","b02cac3cf70080daa6931b6676fa441a"],["E:/GitHubBlog/public/2020/10/22/20201018-1023总结/index.html","584b15b47f4c89b64e75480949a0fe3c"],["E:/GitHubBlog/public/2020/10/24/20201024-1030/index.html","ec0fbc8314b6dd1c02974af8ced17fa9"],["E:/GitHubBlog/public/2020/10/30/1030讨论记录/index.html","db0c4a1151061ba076d374f48c0325ae"],["E:/GitHubBlog/public/2020/11/02/《如何写一个商业计划书》读后感/index.html","88352094931acd72a8eec2a86c180d0d"],["E:/GitHubBlog/public/archives/2020/01/index.html","c748bb9e6e31d3e776ba601f094829c4"],["E:/GitHubBlog/public/archives/2020/02/index.html","f0879feaeb77a29c358a1ff2d9b53884"],["E:/GitHubBlog/public/archives/2020/03/index.html","9c4bca9116f327b646033d4a1f9ded63"],["E:/GitHubBlog/public/archives/2020/04/index.html","98dc147496d9e16656f6342c931c1bc7"],["E:/GitHubBlog/public/archives/2020/05/index.html","b79a472de9377d0a5d06018df7c2fb26"],["E:/GitHubBlog/public/archives/2020/07/index.html","37114007c9247d3335cf3bda7cff4e54"],["E:/GitHubBlog/public/archives/2020/08/index.html","df0719aced9ec7591021c3689c9ee274"],["E:/GitHubBlog/public/archives/2020/09/index.html","619720254f898da4cd068a9ffc85c47b"],["E:/GitHubBlog/public/archives/2020/10/index.html","75d9dd2f0de548a25e52160327310e10"],["E:/GitHubBlog/public/archives/2020/11/index.html","e554df391cb8208456e895a434648710"],["E:/GitHubBlog/public/archives/2020/index.html","d20c73d85ab9bac431c03827a1716f51"],["E:/GitHubBlog/public/archives/2020/page/2/index.html","e3b5e3edc1c15afff4ea54a3aa37251d"],["E:/GitHubBlog/public/archives/2020/page/3/index.html","58723ce75529b61d2a9c1c0e5926ba73"],["E:/GitHubBlog/public/archives/2020/page/4/index.html","391653a48de60bd331d49b0a0fed47b8"],["E:/GitHubBlog/public/archives/2020/page/5/index.html","eb09f88f5c025b92c14551f30dc50fc9"],["E:/GitHubBlog/public/archives/index.html","f0389ec04550f49544552beb67a0f671"],["E:/GitHubBlog/public/archives/page/2/index.html","cfd482bba0fdcd11df370276166ca763"],["E:/GitHubBlog/public/archives/page/3/index.html","cde87348d9e2cb20f0ca61e1f6e46de1"],["E:/GitHubBlog/public/archives/page/4/index.html","8721dda388c1d2837d367b597c6ff99a"],["E:/GitHubBlog/public/archives/page/5/index.html","4ab1470abd4411f95f8f2bb4f37112fd"],["E:/GitHubBlog/public/css/404.css","b1bb50e1fabe41adcec483f6427fb3aa"],["E:/GitHubBlog/public/css/index.css","2389364df340948a5b822422956c2c87"],["E:/GitHubBlog/public/css/var.css","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/img/algolia.svg","fd40b88ac5370a5353a50b8175c1f367"],["E:/GitHubBlog/public/img/avatar.png","6cc4a809d23e3d8946a299ae4ce4e4cd"],["E:/GitHubBlog/public/index.html","83cbe67ca430761a7990708218314fc2"],["E:/GitHubBlog/public/js/copy.js","45aae54bf2e43ac27ecc41eb5e0bacf7"],["E:/GitHubBlog/public/js/fancybox.js","f84d626654b9bbc05720952b3effe062"],["E:/GitHubBlog/public/js/fireworks.js","35933ce61c158ef9c33b5e089fe757ac"],["E:/GitHubBlog/public/js/head.js","347edd99f8e3921b45fa617e839d8182"],["E:/GitHubBlog/public/js/hexo-theme-melody.js","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/js/katex.js","d971ba8b8dee1120ef66744b89cfd2b1"],["E:/GitHubBlog/public/js/scroll.js","a12ad7e37b9e325f8da3d0523d3757c7"],["E:/GitHubBlog/public/js/search/algolia.js","53160985d32d6fd66d98aa0e05b081ac"],["E:/GitHubBlog/public/js/search/local-search.js","1565b508bd866ed6fbd724a3858af5d8"],["E:/GitHubBlog/public/js/sidebar.js","d24db780974e661198eeb2e45f20a28f"],["E:/GitHubBlog/public/js/third-party/anime.min.js","9b4bbe6deb700e1c3606eab732f5eea5"],["E:/GitHubBlog/public/js/third-party/canvas-ribbon.js","09c181544ddff1db701db02ac30453e0"],["E:/GitHubBlog/public/js/third-party/jquery.fancybox.min.js","3c9fa1c1199cd4f874d855ecb1641335"],["E:/GitHubBlog/public/js/third-party/jquery.min.js","c9f5aeeca3ad37bf2aa006139b935f0a"],["E:/GitHubBlog/public/js/third-party/reveal/head.min.js","aad121203010122e05f1766d92385214"],["E:/GitHubBlog/public/js/third-party/reveal/highlight.min.js","44594243bec43813a16371af8fe7e105"],["E:/GitHubBlog/public/js/third-party/reveal/markdown.min.js","7ec4cef5a7fe3f0bf0eb4dc6d7bca114"],["E:/GitHubBlog/public/js/third-party/reveal/marked.min.js","c2a88705e206d71dc21fdc4445349127"],["E:/GitHubBlog/public/js/third-party/reveal/math.min.js","0a278fee2e57c530ab07f7d2d9ea8d96"],["E:/GitHubBlog/public/js/third-party/reveal/notes.min.js","89a0dfae4d706f9c75b317f686c3aa14"],["E:/GitHubBlog/public/js/third-party/reveal/reveal.min.js","8988419d67efb5fe93e291a357e26ec9"],["E:/GitHubBlog/public/js/third-party/reveal/zoom.min.js","9791f96e63e7d534cba2b67d4bda0419"],["E:/GitHubBlog/public/js/third-party/velocity.min.js","64da069aba987ea0512cf610600a56d1"],["E:/GitHubBlog/public/js/third-party/velocity.ui.min.js","c8ca438424a080620f7b2f4ee4b0fff1"],["E:/GitHubBlog/public/js/transition.js","911db4268f0f6621073afcced9e1bfef"],["E:/GitHubBlog/public/js/utils.js","3ff3423d966a1c351e9867813b3f6d36"],["E:/GitHubBlog/public/page/2/index.html","adb6471424ddfb4c215f320e3a96948e"],["E:/GitHubBlog/public/page/3/index.html","a0dea714581c5e83803679a6ffa75de7"],["E:/GitHubBlog/public/page/4/index.html","e5c6775843ed1b8ce373e3912757e317"],["E:/GitHubBlog/public/page/5/index.html","b3966051527f5a7ead433d69416e1345"],["E:/GitHubBlog/public/tags/Android/index.html","7b578248d4fb0ce8c68f53d919a861e6"],["E:/GitHubBlog/public/tags/NLP/index.html","81db08be236df352df8f42bf995ac50b"],["E:/GitHubBlog/public/tags/NLP/page/2/index.html","3a2837da4ae5790dbb0aaa5ea3b40ac8"],["E:/GitHubBlog/public/tags/R/index.html","d6c66f90ecbbdc86daaed94927d225f0"],["E:/GitHubBlog/public/tags/index.html","455e3c88b80a77e52896624f3c910b0b"],["E:/GitHubBlog/public/tags/java/index.html","3bf84c042b52cafc33c03fccb2c6f8cc"],["E:/GitHubBlog/public/tags/leetcode/index.html","ec9d23061ee384e44016c4343669bab2"],["E:/GitHubBlog/public/tags/python/index.html","6379457c686ac08b029c85aad56fde96"],["E:/GitHubBlog/public/tags/总结/index.html","02d24279769e8608bfc9807358ac7f9e"],["E:/GitHubBlog/public/tags/总结/page/2/index.html","8a0fc873ca9c1288a92a93dfb62fec8c"],["E:/GitHubBlog/public/tags/数据分析/index.html","9e098d33be8559a47617ec9797e66531"],["E:/GitHubBlog/public/tags/数据挖掘/index.html","6bbda78cea223fd546f4df7477cdef49"],["E:/GitHubBlog/public/tags/数据结构/index.html","d6dccdb92a34e80883fa23956db4b369"],["E:/GitHubBlog/public/tags/机器学习/index.html","8d9914cef46f0c7cb02d71189efa839f"],["E:/GitHubBlog/public/tags/深度学习/index.html","1dd0e5428eb6cdcbcd85336c2973223e"],["E:/GitHubBlog/public/tags/爬虫/index.html","8c501dedbbc6420442df72a48690d080"],["E:/GitHubBlog/public/tags/笔记/index.html","3727108b73b1c66b85e2e68269a4c04c"],["E:/GitHubBlog/public/tags/论文/index.html","450d843f8a9efa62379c8d0c35fed54a"],["E:/GitHubBlog/public/tags/论文/page/2/index.html","3cd9f0545b0a71bf54afa99bdfddb3c2"],["E:/GitHubBlog/public/tags/读书笔记/index.html","0f871fe65364862b5af0331880882c49"],["E:/GitHubBlog/public/tags/读后感/index.html","3c8753f2328cb71193f4624467cd71fe"]];
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







