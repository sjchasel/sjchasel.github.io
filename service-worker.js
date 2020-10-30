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

var precacheConfig = [["E:/GitHubBlog/public/2020/01/28/hello-world/index.html","88732ff8a93c7312d76dc64633eac37d"],["E:/GitHubBlog/public/2020/02/08/python爬虫学习总结/index.html","22f22acdec96f5a82cccd54f1be12587"],["E:/GitHubBlog/public/2020/02/19/数据结构第一周-绪论/index.html","ca2c40ed92db05a1e52925dc67156f94"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-1/index.html","6b66008e9d3d4fff998a643ab5afed4a"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-2-创建数据集/index.html","f3ccb9641081c19ace01af216aa0e0c6"],["E:/GitHubBlog/public/2020/03/01/《R语言实战》学习总结-3-图形初阶/index.html","d898d0d8a6657c807a4d3aad563beac6"],["E:/GitHubBlog/public/2020/03/02/数据结构第二周-线性结构1/index.html","142d26c28dbb6435a9dcb1911b93a8cd"],["E:/GitHubBlog/public/2020/03/03/《R语言实战》学习总结-4-基本数据管理/index.html","82b98791a63e24c77db8c5bd856e10f2"],["E:/GitHubBlog/public/2020/03/14/数据结构第三周-线性结构2/index.html","e7d2b6a5b7414c3d33057102c9e15bf5"],["E:/GitHubBlog/public/2020/03/14/数据结构第四周-栈与队列/index.html","0d122b1c717ae5d1ce681cf1fc94841e"],["E:/GitHubBlog/public/2020/03/17/leetcode记录-1、88、20/index.html","03c91f86726d025971e5f4b068ed2e1c"],["E:/GitHubBlog/public/2020/03/20/温度转换程序/index.html","d1ec9b5134edb56d03dee863c807288d"],["E:/GitHubBlog/public/2020/03/22/数据结构第五周-递归/index.html","6cdc670ddc52b1ff69729b295fd31777"],["E:/GitHubBlog/public/2020/03/22/数据结构第六周-串与数组/index.html","48c77d191a64dbbbedb9ab32e3c64584"],["E:/GitHubBlog/public/2020/04/02/leetcode记录-52/index.html","f22845cf30d693eec0a7a1a5c3847554"],["E:/GitHubBlog/public/2020/04/17/Android第八周作业-每天第一次打开应用更新数据/index.html","a5a94d398ef58beecb3f72d692247b72"],["E:/GitHubBlog/public/2020/05/21/attention模型/index.html","88b8f3e226713efa3800c74aa1c0ed14"],["E:/GitHubBlog/public/2020/05/21/编码器与解码器框架/index.html","d6483ec1682956a4b9808b985aaa3f26"],["E:/GitHubBlog/public/2020/07/11/思考，快与慢/index.html","f5b023c36d901861cfbe09ac97911810"],["E:/GitHubBlog/public/2020/07/13/20200713-0717总结/index.html","52e63a0a9c3b2746691ace841755c958"],["E:/GitHubBlog/public/2020/07/13/ZeroShotCeres Zero-Shot Relation Extraction/index.html","8246ed137cd2d805a497f0ef542caf0c"],["E:/GitHubBlog/public/2020/07/20/Keyphrase Generation A Text Summarization Struggle/index.html","0ef6caf115219adad4f0dd62db4b5074"],["E:/GitHubBlog/public/2020/07/20/TPR/index.html","37a416d7cd842e25cd565e66916a3005"],["E:/GitHubBlog/public/2020/07/23/Keyphrase Generation for Scientific Document Retrieval/index.html","2c2d64b294cf9543ee01bccaeb611a9a"],["E:/GitHubBlog/public/2020/07/24/20200718-0724总结/index.html","27b6d917413c7672e5133af366dc5ce5"],["E:/GitHubBlog/public/2020/07/31/20200725-0731总结/index.html","d3f122883556d8e921fe10e3892d862c"],["E:/GitHubBlog/public/2020/08/07/20200801-0807总结/index.html","b48329f20ede1762939fdf73e79677ee"],["E:/GitHubBlog/public/2020/08/12/逻辑回归算法/index.html","783c3fbc88d6a428e8cdd917d5e8b6fa"],["E:/GitHubBlog/public/2020/08/13/李宏毅-机器学习21/index.html","e8e5521444fcd5fc20ef6989219863ef"],["E:/GitHubBlog/public/2020/08/14/20200808-0815总结/index.html","bc5ee10555ed13ccfee538f673aab5af"],["E:/GitHubBlog/public/2020/09/03/20200816-0903总结/index.html","f6bdac6a382713f11d0a10ef8e3f8b46"],["E:/GitHubBlog/public/2020/09/16/20200914-0918总结/index.html","2e9837896ecb79e09ebc38f158312a3d"],["E:/GitHubBlog/public/2020/09/25/20200919-0925总结/index.html","e213d9f81d7c686a809ef90111ed9a13"],["E:/GitHubBlog/public/2020/10/06/《Deep Keyphrase Generation》详细解析！/index.html","fe47bb0bf41cb2a16249db04e7706fee"],["E:/GitHubBlog/public/2020/10/16/20201008-1017总结/index.html","61cb65588d74b41b335da2ff8910afe7"],["E:/GitHubBlog/public/2020/10/20/特征选择的几种方法(一)————Filter过滤法/index.html","8099d3d48b41ada9e9afb00accd57b94"],["E:/GitHubBlog/public/2020/10/21/PCA/index.html","ce72021286c3d7a4d02754ba3e6fe2a5"],["E:/GitHubBlog/public/2020/10/22/20201018-1023总结/index.html","dc9904a84a53cb899c70627d87dcd078"],["E:/GitHubBlog/public/2020/10/24/20201024-1030/index.html","654971f80e82b00bc957568038693e41"],["E:/GitHubBlog/public/2020/10/30/1030讨论记录/index.html","7eb552a146c6312b0dc69cbcfb93ac19"],["E:/GitHubBlog/public/archives/2020/01/index.html","63b43ef55da130ad05b8892f66917a99"],["E:/GitHubBlog/public/archives/2020/02/index.html","cee9d561c4661b97039e163dec3f35a7"],["E:/GitHubBlog/public/archives/2020/03/index.html","c2bbc07de5795c6093c95ce0c783919f"],["E:/GitHubBlog/public/archives/2020/04/index.html","c2e1d677465a662d48ae6a59218b0cb7"],["E:/GitHubBlog/public/archives/2020/05/index.html","7031b1d39843bb25035fbc7ea207d910"],["E:/GitHubBlog/public/archives/2020/07/index.html","6981283b1a4811f565f87f4c071e332f"],["E:/GitHubBlog/public/archives/2020/08/index.html","c3d784e7a3d962ad3bc24789a54a84d5"],["E:/GitHubBlog/public/archives/2020/09/index.html","08afc5bd0fecab144da340eaba5fdfe2"],["E:/GitHubBlog/public/archives/2020/10/index.html","2c8ff91193ff22c72eb5e6d42c9b4b5e"],["E:/GitHubBlog/public/archives/2020/index.html","f10776bcf4d5a583f2971d62c0d2708d"],["E:/GitHubBlog/public/archives/2020/page/2/index.html","2fb4922a110628c6e4a5b8fda6a3e082"],["E:/GitHubBlog/public/archives/2020/page/3/index.html","6d310b11e1750c098427e1d367a091ee"],["E:/GitHubBlog/public/archives/2020/page/4/index.html","8eb2af3a176839f587416f1e2316b241"],["E:/GitHubBlog/public/archives/index.html","5dc9a128d1d43ed2bcb9c4dd6ed14747"],["E:/GitHubBlog/public/archives/page/2/index.html","82bf3e4a5e2af0ec26c97cd7c8a92ab5"],["E:/GitHubBlog/public/archives/page/3/index.html","3a46efa5bdfcd3b2af2ee7ca158a6877"],["E:/GitHubBlog/public/archives/page/4/index.html","0d5bc2073e73b84df265891edc69f6f9"],["E:/GitHubBlog/public/css/404.css","b1bb50e1fabe41adcec483f6427fb3aa"],["E:/GitHubBlog/public/css/index.css","2389364df340948a5b822422956c2c87"],["E:/GitHubBlog/public/css/var.css","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/img/algolia.svg","fd40b88ac5370a5353a50b8175c1f367"],["E:/GitHubBlog/public/img/avatar.png","6cc4a809d23e3d8946a299ae4ce4e4cd"],["E:/GitHubBlog/public/index.html","265b31d6ddd7b297363ec27efce0b373"],["E:/GitHubBlog/public/js/copy.js","45aae54bf2e43ac27ecc41eb5e0bacf7"],["E:/GitHubBlog/public/js/fancybox.js","f84d626654b9bbc05720952b3effe062"],["E:/GitHubBlog/public/js/fireworks.js","35933ce61c158ef9c33b5e089fe757ac"],["E:/GitHubBlog/public/js/head.js","347edd99f8e3921b45fa617e839d8182"],["E:/GitHubBlog/public/js/hexo-theme-melody.js","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/js/katex.js","d971ba8b8dee1120ef66744b89cfd2b1"],["E:/GitHubBlog/public/js/scroll.js","a12ad7e37b9e325f8da3d0523d3757c7"],["E:/GitHubBlog/public/js/search/algolia.js","53160985d32d6fd66d98aa0e05b081ac"],["E:/GitHubBlog/public/js/search/local-search.js","1565b508bd866ed6fbd724a3858af5d8"],["E:/GitHubBlog/public/js/sidebar.js","d24db780974e661198eeb2e45f20a28f"],["E:/GitHubBlog/public/js/third-party/anime.min.js","9b4bbe6deb700e1c3606eab732f5eea5"],["E:/GitHubBlog/public/js/third-party/canvas-ribbon.js","09c181544ddff1db701db02ac30453e0"],["E:/GitHubBlog/public/js/third-party/jquery.fancybox.min.js","3c9fa1c1199cd4f874d855ecb1641335"],["E:/GitHubBlog/public/js/third-party/jquery.min.js","c9f5aeeca3ad37bf2aa006139b935f0a"],["E:/GitHubBlog/public/js/third-party/reveal/head.min.js","aad121203010122e05f1766d92385214"],["E:/GitHubBlog/public/js/third-party/reveal/highlight.min.js","44594243bec43813a16371af8fe7e105"],["E:/GitHubBlog/public/js/third-party/reveal/markdown.min.js","7ec4cef5a7fe3f0bf0eb4dc6d7bca114"],["E:/GitHubBlog/public/js/third-party/reveal/marked.min.js","c2a88705e206d71dc21fdc4445349127"],["E:/GitHubBlog/public/js/third-party/reveal/math.min.js","0a278fee2e57c530ab07f7d2d9ea8d96"],["E:/GitHubBlog/public/js/third-party/reveal/notes.min.js","89a0dfae4d706f9c75b317f686c3aa14"],["E:/GitHubBlog/public/js/third-party/reveal/reveal.min.js","8988419d67efb5fe93e291a357e26ec9"],["E:/GitHubBlog/public/js/third-party/reveal/zoom.min.js","9791f96e63e7d534cba2b67d4bda0419"],["E:/GitHubBlog/public/js/third-party/velocity.min.js","64da069aba987ea0512cf610600a56d1"],["E:/GitHubBlog/public/js/third-party/velocity.ui.min.js","c8ca438424a080620f7b2f4ee4b0fff1"],["E:/GitHubBlog/public/js/transition.js","911db4268f0f6621073afcced9e1bfef"],["E:/GitHubBlog/public/js/utils.js","3ff3423d966a1c351e9867813b3f6d36"],["E:/GitHubBlog/public/page/2/index.html","6323c9cf5281ae9c3f14b381c0d79987"],["E:/GitHubBlog/public/page/3/index.html","27bf533c5fe201a12c28af4f469701f1"],["E:/GitHubBlog/public/page/4/index.html","82bc88b6c2ffdf327db299cdcd381be4"],["E:/GitHubBlog/public/tags/Android/index.html","bb0d780e2b511e6063c318b000d83b95"],["E:/GitHubBlog/public/tags/NLP/index.html","ad6a8df4d67af30ba443a6808d2552e3"],["E:/GitHubBlog/public/tags/NLP/page/2/index.html","9390e8765685961994c0010f77267217"],["E:/GitHubBlog/public/tags/R/index.html","8f43693e079cc2712c494e5684439c1a"],["E:/GitHubBlog/public/tags/index.html","705ff96d667a9f3f1203e17c22de2aed"],["E:/GitHubBlog/public/tags/java/index.html","fea1fa8e235a0549d9ff02a959cb3437"],["E:/GitHubBlog/public/tags/leetcode/index.html","4d426760806f9456f6dce6ffbd21cb09"],["E:/GitHubBlog/public/tags/python/index.html","f20a9f5b4709f484c4c9a6d6216d6d2e"],["E:/GitHubBlog/public/tags/总结/index.html","e286ee67ad66d2237e9e237b22250de0"],["E:/GitHubBlog/public/tags/总结/page/2/index.html","17a98954958a303487208f9171173769"],["E:/GitHubBlog/public/tags/数据分析/index.html","f2eba35c303e8e228e295d6702307b8c"],["E:/GitHubBlog/public/tags/数据挖掘/index.html","f2cff0c881bf59b61050ea6130edf89b"],["E:/GitHubBlog/public/tags/数据结构/index.html","67cad56e6bb3d389c20cb21d9c679314"],["E:/GitHubBlog/public/tags/机器学习/index.html","52278781fa62676b25f44a2c117bc732"],["E:/GitHubBlog/public/tags/深度学习/index.html","fa3ed3c1ad32550102f716cd3b253790"],["E:/GitHubBlog/public/tags/爬虫/index.html","e5046a94c3a3ae8fdd98ba4e4846f9f8"],["E:/GitHubBlog/public/tags/笔记/index.html","b33911ddcd85c0f4b5f8bcba4be8b051"],["E:/GitHubBlog/public/tags/论文/index.html","9bc2e21115263afb798ddb086d58c135"],["E:/GitHubBlog/public/tags/论文/page/2/index.html","e9bb21f2570e37445a205f98dc59c4df"],["E:/GitHubBlog/public/tags/读书笔记/index.html","5b3582c032ef55c1e412b26b4a65aff1"]];
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







