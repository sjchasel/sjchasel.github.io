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

var precacheConfig = [["E:/GitHubBlog/public/2020/01/28/hello-world/index.html","60d709e560701baa9b83e4351577e164"],["E:/GitHubBlog/public/2020/02/08/python爬虫学习总结/index.html","ec84436b0540c91a5f0a34219fcc4bf0"],["E:/GitHubBlog/public/2020/02/19/数据结构第一周-绪论/index.html","19569dd5d5520043ed513b8d2a840f0d"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-1/index.html","45b592068be75307f32c1b86835aaa47"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-2-创建数据集/index.html","05707bc33abef6f6aeda73870b570442"],["E:/GitHubBlog/public/2020/03/01/《R语言实战》学习总结-3-图形初阶/index.html","b3cef0b2f1c45fcb6869fcdc6c3294bb"],["E:/GitHubBlog/public/2020/03/02/数据结构第二周-线性结构1/index.html","b7351a94cd4f66f2f3ad1a3f7809d240"],["E:/GitHubBlog/public/2020/03/03/《R语言实战》学习总结-4-基本数据管理/index.html","ae0b71f0b1b1dcc098c262364a1de28f"],["E:/GitHubBlog/public/2020/03/14/数据结构第三周-线性结构2/index.html","a751cb2c8a46e78ba3067898e408ed55"],["E:/GitHubBlog/public/2020/03/14/数据结构第四周-栈与队列/index.html","bdacaa33ab5f1065b06da6660047d607"],["E:/GitHubBlog/public/2020/03/17/leetcode记录-1、88、20/index.html","61a20f4b20ca004b9540ae38f927146a"],["E:/GitHubBlog/public/2020/03/20/温度转换程序/index.html","2364babd81703ef6f04f15b333abe6a0"],["E:/GitHubBlog/public/2020/03/22/数据结构第五周-递归/index.html","8476587e8443ef0998b764d51c257cf8"],["E:/GitHubBlog/public/2020/03/22/数据结构第六周-串与数组/index.html","1953b5b22a2fc0ecb2b92a0e8dbabf14"],["E:/GitHubBlog/public/2020/04/02/leetcode记录-52/index.html","b7cad36f758e39590211ce788d429cb4"],["E:/GitHubBlog/public/2020/04/17/Android第八周作业-每天第一次打开应用更新数据/index.html","a4f943fb491ef04268ec29f36a4ce5e1"],["E:/GitHubBlog/public/2020/05/21/attention模型/index.html","4aaf37d0fba6b80fa30ab4ca70087cb2"],["E:/GitHubBlog/public/2020/05/21/编码器与解码器框架/index.html","72af37de022f315883c5f2914b5dd2b8"],["E:/GitHubBlog/public/2020/07/11/思考，快与慢/index.html","835df671e5d843cd9270570978a6e545"],["E:/GitHubBlog/public/2020/07/13/20200713-0717总结/index.html","a65ada15fb063c8b48712990ebfda3e5"],["E:/GitHubBlog/public/2020/07/13/ZeroShotCeres Zero-Shot Relation Extraction/index.html","dea14c1fb0ac5e4ee0abf0eb9ab964a8"],["E:/GitHubBlog/public/2020/07/20/Keyphrase Generation A Text Summarization Struggle/index.html","3789cbbb009d1b8c0779d4083da41801"],["E:/GitHubBlog/public/2020/07/20/TPR/index.html","b90dc4963a58aeefa8b1444130c6333c"],["E:/GitHubBlog/public/2020/07/23/Keyphrase Generation for Scientific Document Retrieval/index.html","cd519ff4da1ec2965ae5d491314abea9"],["E:/GitHubBlog/public/2020/07/24/20200718-0724总结/index.html","d0799df1c5eea46c4733de5a40c03034"],["E:/GitHubBlog/public/2020/07/31/20200725-0731总结/index.html","2cdf2ee4f5b1bc1462ad9a5ad3c9b95d"],["E:/GitHubBlog/public/2020/08/07/20200801-0807总结/index.html","40daaf375a1077c9a79ad23058be1d0b"],["E:/GitHubBlog/public/2020/08/12/逻辑回归算法/index.html","fe3a185cbe3a88ff11ea2fc69c48fd6d"],["E:/GitHubBlog/public/2020/08/13/李宏毅-机器学习21/index.html","1de8176946428bb1ed255a062235b13b"],["E:/GitHubBlog/public/2020/08/14/20200808-0815总结/index.html","c504f908e033c89e62d25dd4f2127fc1"],["E:/GitHubBlog/public/2020/09/03/20200816-0903总结/index.html","0edba0f77f3906edea1f216e4c360823"],["E:/GitHubBlog/public/archives/2020/01/index.html","6eaae0109edc79b41ccfed52de62c628"],["E:/GitHubBlog/public/archives/2020/02/index.html","99eaad8c1b57f823124821c22da2653b"],["E:/GitHubBlog/public/archives/2020/03/index.html","25c37a032a9738e63afd3a5e0f7c9c6d"],["E:/GitHubBlog/public/archives/2020/04/index.html","0e9143896ff509e6217098be63a41d3e"],["E:/GitHubBlog/public/archives/2020/05/index.html","5d203ad1e3d7944f4be0c6d410e3c87d"],["E:/GitHubBlog/public/archives/2020/07/index.html","d8de344920512ad324c119ba53a07a03"],["E:/GitHubBlog/public/archives/2020/08/index.html","b3d175b50d884b35d5050520ce078050"],["E:/GitHubBlog/public/archives/2020/09/index.html","377ef16eae4c0a1b9384b4c6f1b8c1d8"],["E:/GitHubBlog/public/archives/2020/index.html","d593115dda8828d69fa167b9e18b5fd5"],["E:/GitHubBlog/public/archives/2020/page/2/index.html","e92087951d868aa114b26dc39b0bd175"],["E:/GitHubBlog/public/archives/2020/page/3/index.html","fa6588b1d190c62db4a76186767b3539"],["E:/GitHubBlog/public/archives/2020/page/4/index.html","788f41b15776a91b30b4f1727f54ba61"],["E:/GitHubBlog/public/archives/index.html","5be1bfe661f1c62cb9215eccc96fcc25"],["E:/GitHubBlog/public/archives/page/2/index.html","6f1b592d2c530d8e53bdd1b3a0ef7266"],["E:/GitHubBlog/public/archives/page/3/index.html","79a6ad8fb7c676df8300695717daaefe"],["E:/GitHubBlog/public/archives/page/4/index.html","14b68585c2d1c342c426c02d693d8eb3"],["E:/GitHubBlog/public/css/404.css","b1bb50e1fabe41adcec483f6427fb3aa"],["E:/GitHubBlog/public/css/index.css","2389364df340948a5b822422956c2c87"],["E:/GitHubBlog/public/css/var.css","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/img/algolia.svg","fd40b88ac5370a5353a50b8175c1f367"],["E:/GitHubBlog/public/img/avatar.png","6cc4a809d23e3d8946a299ae4ce4e4cd"],["E:/GitHubBlog/public/index.html","c9bf625270c65a0f781c3e464f51aaaa"],["E:/GitHubBlog/public/js/copy.js","45aae54bf2e43ac27ecc41eb5e0bacf7"],["E:/GitHubBlog/public/js/fancybox.js","f84d626654b9bbc05720952b3effe062"],["E:/GitHubBlog/public/js/fireworks.js","35933ce61c158ef9c33b5e089fe757ac"],["E:/GitHubBlog/public/js/head.js","347edd99f8e3921b45fa617e839d8182"],["E:/GitHubBlog/public/js/hexo-theme-melody.js","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/js/katex.js","d971ba8b8dee1120ef66744b89cfd2b1"],["E:/GitHubBlog/public/js/scroll.js","a12ad7e37b9e325f8da3d0523d3757c7"],["E:/GitHubBlog/public/js/search/algolia.js","53160985d32d6fd66d98aa0e05b081ac"],["E:/GitHubBlog/public/js/search/local-search.js","1565b508bd866ed6fbd724a3858af5d8"],["E:/GitHubBlog/public/js/sidebar.js","d24db780974e661198eeb2e45f20a28f"],["E:/GitHubBlog/public/js/third-party/anime.min.js","9b4bbe6deb700e1c3606eab732f5eea5"],["E:/GitHubBlog/public/js/third-party/canvas-ribbon.js","09c181544ddff1db701db02ac30453e0"],["E:/GitHubBlog/public/js/third-party/jquery.fancybox.min.js","3c9fa1c1199cd4f874d855ecb1641335"],["E:/GitHubBlog/public/js/third-party/jquery.min.js","c9f5aeeca3ad37bf2aa006139b935f0a"],["E:/GitHubBlog/public/js/third-party/reveal/head.min.js","aad121203010122e05f1766d92385214"],["E:/GitHubBlog/public/js/third-party/reveal/highlight.min.js","44594243bec43813a16371af8fe7e105"],["E:/GitHubBlog/public/js/third-party/reveal/markdown.min.js","7ec4cef5a7fe3f0bf0eb4dc6d7bca114"],["E:/GitHubBlog/public/js/third-party/reveal/marked.min.js","c2a88705e206d71dc21fdc4445349127"],["E:/GitHubBlog/public/js/third-party/reveal/math.min.js","0a278fee2e57c530ab07f7d2d9ea8d96"],["E:/GitHubBlog/public/js/third-party/reveal/notes.min.js","89a0dfae4d706f9c75b317f686c3aa14"],["E:/GitHubBlog/public/js/third-party/reveal/reveal.min.js","8988419d67efb5fe93e291a357e26ec9"],["E:/GitHubBlog/public/js/third-party/reveal/zoom.min.js","9791f96e63e7d534cba2b67d4bda0419"],["E:/GitHubBlog/public/js/third-party/velocity.min.js","64da069aba987ea0512cf610600a56d1"],["E:/GitHubBlog/public/js/third-party/velocity.ui.min.js","c8ca438424a080620f7b2f4ee4b0fff1"],["E:/GitHubBlog/public/js/transition.js","911db4268f0f6621073afcced9e1bfef"],["E:/GitHubBlog/public/js/utils.js","3ff3423d966a1c351e9867813b3f6d36"],["E:/GitHubBlog/public/page/2/index.html","f411c2381a8a78ad21e931f35ac64b18"],["E:/GitHubBlog/public/page/3/index.html","7417138ad365b301e501291d6d51b28f"],["E:/GitHubBlog/public/page/4/index.html","fdd7f97bd4d485f44d2d31855a84f109"],["E:/GitHubBlog/public/tags/Android/index.html","88da7bd203e9bd855b882558524e9473"],["E:/GitHubBlog/public/tags/NLP/index.html","61c3fc556fa55045674fada86f3169f2"],["E:/GitHubBlog/public/tags/R/index.html","885799525c667f43fc80b99ff8720178"],["E:/GitHubBlog/public/tags/index.html","89e0fbcb5a87175062a05753dfb304a5"],["E:/GitHubBlog/public/tags/java/index.html","251e3522a03db348c28b950af536b022"],["E:/GitHubBlog/public/tags/leetcode/index.html","303e557d765945b4c2a92c32eff82d0f"],["E:/GitHubBlog/public/tags/python/index.html","c05d054c349be896d0c63237b3c4a3e9"],["E:/GitHubBlog/public/tags/总结/index.html","0e92ecdd5c97c60facb306437c786ed2"],["E:/GitHubBlog/public/tags/数据分析/index.html","df9de5dd9dfe7a407000739a6325c0db"],["E:/GitHubBlog/public/tags/数据结构/index.html","d904de3ae286438ee57f970166b7e110"],["E:/GitHubBlog/public/tags/机器学习/index.html","04d45634970eadf4b3bdc7c238c4041d"],["E:/GitHubBlog/public/tags/深度学习/index.html","fedc7e25220679c5294456416ad10d3f"],["E:/GitHubBlog/public/tags/爬虫/index.html","ee806318e8a7d6ed8a51f13280019d05"],["E:/GitHubBlog/public/tags/笔记/index.html","8a1e0ee95b359dee0ba3d6ea67c3d41a"],["E:/GitHubBlog/public/tags/论文/index.html","c0cbc84b359b2d21fdfc127bcdfde9d6"],["E:/GitHubBlog/public/tags/论文/page/2/index.html","bf8ebfcc66e37b7a5e7650d79cfabccf"],["E:/GitHubBlog/public/tags/读书笔记/index.html","380182a2bad2cf7901bc4a98729e1271"]];
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







