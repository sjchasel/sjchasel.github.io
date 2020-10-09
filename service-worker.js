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

var precacheConfig = [["E:/GitHubBlog/public/2020/01/28/hello-world/index.html","94b182cf4baca3537a773206841bfff2"],["E:/GitHubBlog/public/2020/02/08/python爬虫学习总结/index.html","9909ff1110b672e3732a69f88ee0d8db"],["E:/GitHubBlog/public/2020/02/19/数据结构第一周-绪论/index.html","7d5cbeabf6bdd96019c98323cc58615d"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-1/index.html","0bb8f36cc81310711a136ee8c74c9b31"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-2-创建数据集/index.html","e15d874217bfba1292372b7f09da812c"],["E:/GitHubBlog/public/2020/03/01/《R语言实战》学习总结-3-图形初阶/index.html","627a79cd1ffba588166994257a60907b"],["E:/GitHubBlog/public/2020/03/02/数据结构第二周-线性结构1/index.html","2eea57169f369fc4c31f957d8e329e68"],["E:/GitHubBlog/public/2020/03/03/《R语言实战》学习总结-4-基本数据管理/index.html","57b19b662b6dbe4144cbae01575e9262"],["E:/GitHubBlog/public/2020/03/14/数据结构第三周-线性结构2/index.html","78fa54f6ccb45e16a7823679386d9750"],["E:/GitHubBlog/public/2020/03/14/数据结构第四周-栈与队列/index.html","615f7471933aed395e1ddf082802a8ec"],["E:/GitHubBlog/public/2020/03/17/leetcode记录-1、88、20/index.html","0e799f810027fea72f197e3dd6ba1608"],["E:/GitHubBlog/public/2020/03/20/温度转换程序/index.html","63841f82ff6df0cf13970eeace8872b0"],["E:/GitHubBlog/public/2020/03/22/数据结构第五周-递归/index.html","2ab7eb0ac4178a006251db3edb49659a"],["E:/GitHubBlog/public/2020/03/22/数据结构第六周-串与数组/index.html","94b2af486f6533b83fc23672612b48d4"],["E:/GitHubBlog/public/2020/04/02/leetcode记录-52/index.html","e52434dfa4572089e04022c51dc44cf2"],["E:/GitHubBlog/public/2020/04/17/Android第八周作业-每天第一次打开应用更新数据/index.html","c1f6738b70ecc016f7468367ccaec8ff"],["E:/GitHubBlog/public/2020/05/21/attention模型/index.html","fb8cca481d03112bf376fcddd1100bf2"],["E:/GitHubBlog/public/2020/05/21/编码器与解码器框架/index.html","a1bc0dc29755b9c661314fec8db65fc7"],["E:/GitHubBlog/public/2020/07/11/思考，快与慢/index.html","a3740ba4979833e9421271522945d1a5"],["E:/GitHubBlog/public/2020/07/13/20200713-0717总结/index.html","b28149550cd569cd03a9b0414a708110"],["E:/GitHubBlog/public/2020/07/13/ZeroShotCeres Zero-Shot Relation Extraction/index.html","af510a82eef3b971302970532877feeb"],["E:/GitHubBlog/public/2020/07/20/Keyphrase Generation A Text Summarization Struggle/index.html","b490794e48437a7db7fc6c565615fbc6"],["E:/GitHubBlog/public/2020/07/20/TPR/index.html","5cf3157d5355187aaee0734be8fe45a5"],["E:/GitHubBlog/public/2020/07/23/Keyphrase Generation for Scientific Document Retrieval/index.html","c82d1939be207b3703619d63f8b1cd52"],["E:/GitHubBlog/public/2020/07/24/20200718-0724总结/index.html","d2bef8eae948ad2b8fd1ae50debc84b6"],["E:/GitHubBlog/public/2020/07/31/20200725-0731总结/index.html","fc4b404e51a5d5e48e6385e7d1ceb2e1"],["E:/GitHubBlog/public/2020/08/07/20200801-0807总结/index.html","75aeb04db351f990ff4cce88466a3b92"],["E:/GitHubBlog/public/2020/08/12/逻辑回归算法/index.html","b2f75d0615c2d1b893421ee5737cd04c"],["E:/GitHubBlog/public/2020/08/13/李宏毅-机器学习21/index.html","a18d2bbaec9b82c50c740c6a964c615d"],["E:/GitHubBlog/public/2020/08/14/20200808-0815总结/index.html","26c7ca40bd93dbffc1a288bd02ecbe0c"],["E:/GitHubBlog/public/2020/09/03/20200816-0903总结/index.html","96e695fd742550e488239177a7d2660e"],["E:/GitHubBlog/public/2020/09/16/20200914-0918总结/index.html","b76ad65cc38597567de02fd0f9f1d608"],["E:/GitHubBlog/public/2020/09/25/20200919-0925总结/index.html","a7578a3d8b9fbb8a04b606c597504401"],["E:/GitHubBlog/public/2020/10/06/《Deep Keyphrase Generation》详细解析！/index.html","15bcd364c21ec3097a4499a24c53723c"],["E:/GitHubBlog/public/archives/2020/01/index.html","7fc025b43e8e60318022f89ad3cd8d52"],["E:/GitHubBlog/public/archives/2020/02/index.html","ea4517048c4992e69674a601abf0139f"],["E:/GitHubBlog/public/archives/2020/03/index.html","02a6e40a1b42f3bf337454ce3a72b756"],["E:/GitHubBlog/public/archives/2020/04/index.html","f23b1509816e9ffeab1d614aecc36c87"],["E:/GitHubBlog/public/archives/2020/05/index.html","6aded5915cb8cda27ca26c81af5065ca"],["E:/GitHubBlog/public/archives/2020/07/index.html","fcd6bacc047a45f52effe49dfbc6c0cb"],["E:/GitHubBlog/public/archives/2020/08/index.html","5a1cee6f1c14ebbcd46ec38c999133eb"],["E:/GitHubBlog/public/archives/2020/09/index.html","e54871f3203f8bd3ecd0a9cc3ebdb426"],["E:/GitHubBlog/public/archives/2020/10/index.html","d9856acd84631ddb3f0d23e305ad6da4"],["E:/GitHubBlog/public/archives/2020/index.html","da4193e405780eab3e02979042a8e956"],["E:/GitHubBlog/public/archives/2020/page/2/index.html","68d691fae8e976bc0a6ed9da291ace71"],["E:/GitHubBlog/public/archives/2020/page/3/index.html","1fbb590620d325e2ef5e583365ccf834"],["E:/GitHubBlog/public/archives/2020/page/4/index.html","86cf1bb0e85beae5388d684accb90c01"],["E:/GitHubBlog/public/archives/index.html","02adf8e0a5e82936bdf7604113d5e27c"],["E:/GitHubBlog/public/archives/page/2/index.html","bf2bad718718076686601a2d5fc0b011"],["E:/GitHubBlog/public/archives/page/3/index.html","8abe9fc79ac33fce513c7810759aa61b"],["E:/GitHubBlog/public/archives/page/4/index.html","caea066f3c5fc8181f66743a44065a19"],["E:/GitHubBlog/public/css/404.css","b1bb50e1fabe41adcec483f6427fb3aa"],["E:/GitHubBlog/public/css/index.css","2389364df340948a5b822422956c2c87"],["E:/GitHubBlog/public/css/var.css","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/img/algolia.svg","fd40b88ac5370a5353a50b8175c1f367"],["E:/GitHubBlog/public/img/avatar.png","6cc4a809d23e3d8946a299ae4ce4e4cd"],["E:/GitHubBlog/public/index.html","f24ec3cb8222880e36e9c9924bfaf5a3"],["E:/GitHubBlog/public/js/copy.js","45aae54bf2e43ac27ecc41eb5e0bacf7"],["E:/GitHubBlog/public/js/fancybox.js","f84d626654b9bbc05720952b3effe062"],["E:/GitHubBlog/public/js/fireworks.js","35933ce61c158ef9c33b5e089fe757ac"],["E:/GitHubBlog/public/js/head.js","347edd99f8e3921b45fa617e839d8182"],["E:/GitHubBlog/public/js/hexo-theme-melody.js","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/js/katex.js","d971ba8b8dee1120ef66744b89cfd2b1"],["E:/GitHubBlog/public/js/scroll.js","a12ad7e37b9e325f8da3d0523d3757c7"],["E:/GitHubBlog/public/js/search/algolia.js","53160985d32d6fd66d98aa0e05b081ac"],["E:/GitHubBlog/public/js/search/local-search.js","1565b508bd866ed6fbd724a3858af5d8"],["E:/GitHubBlog/public/js/sidebar.js","d24db780974e661198eeb2e45f20a28f"],["E:/GitHubBlog/public/js/third-party/anime.min.js","9b4bbe6deb700e1c3606eab732f5eea5"],["E:/GitHubBlog/public/js/third-party/canvas-ribbon.js","09c181544ddff1db701db02ac30453e0"],["E:/GitHubBlog/public/js/third-party/jquery.fancybox.min.js","3c9fa1c1199cd4f874d855ecb1641335"],["E:/GitHubBlog/public/js/third-party/jquery.min.js","c9f5aeeca3ad37bf2aa006139b935f0a"],["E:/GitHubBlog/public/js/third-party/reveal/head.min.js","aad121203010122e05f1766d92385214"],["E:/GitHubBlog/public/js/third-party/reveal/highlight.min.js","44594243bec43813a16371af8fe7e105"],["E:/GitHubBlog/public/js/third-party/reveal/markdown.min.js","7ec4cef5a7fe3f0bf0eb4dc6d7bca114"],["E:/GitHubBlog/public/js/third-party/reveal/marked.min.js","c2a88705e206d71dc21fdc4445349127"],["E:/GitHubBlog/public/js/third-party/reveal/math.min.js","0a278fee2e57c530ab07f7d2d9ea8d96"],["E:/GitHubBlog/public/js/third-party/reveal/notes.min.js","89a0dfae4d706f9c75b317f686c3aa14"],["E:/GitHubBlog/public/js/third-party/reveal/reveal.min.js","8988419d67efb5fe93e291a357e26ec9"],["E:/GitHubBlog/public/js/third-party/reveal/zoom.min.js","9791f96e63e7d534cba2b67d4bda0419"],["E:/GitHubBlog/public/js/third-party/velocity.min.js","64da069aba987ea0512cf610600a56d1"],["E:/GitHubBlog/public/js/third-party/velocity.ui.min.js","c8ca438424a080620f7b2f4ee4b0fff1"],["E:/GitHubBlog/public/js/transition.js","911db4268f0f6621073afcced9e1bfef"],["E:/GitHubBlog/public/js/utils.js","3ff3423d966a1c351e9867813b3f6d36"],["E:/GitHubBlog/public/page/2/index.html","00ecbc8e9166d8c14d5874fa20df2872"],["E:/GitHubBlog/public/page/3/index.html","c132fb8e48ce16cb3bfc3e4bf59bcd34"],["E:/GitHubBlog/public/page/4/index.html","6a055021d12c6e075b60184bd2424dde"],["E:/GitHubBlog/public/tags/Android/index.html","174cd6c54f5cc79a7518230ae46dbd46"],["E:/GitHubBlog/public/tags/NLP/index.html","cbf303b383503951e8ca4d46bb4ea441"],["E:/GitHubBlog/public/tags/NLP/page/2/index.html","3474262c29691bfdb9d258b2cdd89e0a"],["E:/GitHubBlog/public/tags/R/index.html","f7fbd6b457f92626f9e60f4d9c2150ad"],["E:/GitHubBlog/public/tags/index.html","60dfc03c5afbd24862c1b7fa94d604da"],["E:/GitHubBlog/public/tags/java/index.html","9e4649d828d3fcd6a21e1bbe6ed69c0f"],["E:/GitHubBlog/public/tags/leetcode/index.html","09dbced91a43705ccf128bcc46613e4f"],["E:/GitHubBlog/public/tags/python/index.html","c60f82033e734a3f62f1fa24baf89723"],["E:/GitHubBlog/public/tags/总结/index.html","654b3a7c7509f68a9ff35f71efb18a4e"],["E:/GitHubBlog/public/tags/数据分析/index.html","d288e26f04f2e704e3457fe723784409"],["E:/GitHubBlog/public/tags/数据结构/index.html","dc241e60fef0fca08b6f0b8b8773e031"],["E:/GitHubBlog/public/tags/机器学习/index.html","11f9f2e507711291f5df71cb301c54d1"],["E:/GitHubBlog/public/tags/深度学习/index.html","aeb4979718875634d1b2dbcb1ffa89c3"],["E:/GitHubBlog/public/tags/爬虫/index.html","9eb14b076cb11e0b69c603d6979d28db"],["E:/GitHubBlog/public/tags/笔记/index.html","5e0f42cc85da012d3006534b02ac8529"],["E:/GitHubBlog/public/tags/论文/index.html","17a6f812f6bcf973735eab764f7370a9"],["E:/GitHubBlog/public/tags/论文/page/2/index.html","c8d9ffe1b71ecc24f8aa7e6052924e74"],["E:/GitHubBlog/public/tags/读书笔记/index.html","6741e5d47f143cb0dbbdd473d1882b43"]];
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







