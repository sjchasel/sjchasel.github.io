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

var precacheConfig = [["E:/GitHubBlog/public/2020/01/28/hello-world/index.html","0f2d3e8173d5dd7d742db3ea90bcbe6e"],["E:/GitHubBlog/public/2020/02/08/python爬虫学习总结/index.html","d2c5956c3c0e7a5fe21e40f8280760f8"],["E:/GitHubBlog/public/2020/02/19/数据结构第一周-绪论/index.html","81bfa3ec1c6a407c2512b938f320cead"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-1/index.html","3fe0a7f0086a97dadd98ab84864eae9b"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-2-创建数据集/index.html","65ea66d4caef98f65a474f6af2ddf92b"],["E:/GitHubBlog/public/2020/03/01/《R语言实战》学习总结-3-图形初阶/index.html","45882b65c752a5fae34c2752670eb6b8"],["E:/GitHubBlog/public/2020/03/02/数据结构第二周-线性结构1/index.html","9cb1f0b55bbfb556f2a5a5843f798f35"],["E:/GitHubBlog/public/2020/03/03/《R语言实战》学习总结-4-基本数据管理/index.html","84588285befcae736c1495e36d299066"],["E:/GitHubBlog/public/2020/03/14/数据结构第三周-线性结构2/index.html","6c675ee70bf56cfff9a50a28a83c2da0"],["E:/GitHubBlog/public/2020/03/14/数据结构第四周-栈与队列/index.html","d72ba60b248fa5dd3d8111c9d4392131"],["E:/GitHubBlog/public/2020/03/17/leetcode记录-1、88、20/index.html","55bb18eb5f648870705d205b18e00d50"],["E:/GitHubBlog/public/2020/03/20/温度转换程序/index.html","99319ad757b8861e95d2eb0c0679191c"],["E:/GitHubBlog/public/2020/03/22/数据结构第五周-递归/index.html","6abca884a4c40c86c53eac2e8e1c2cda"],["E:/GitHubBlog/public/2020/03/22/数据结构第六周-串与数组/index.html","8843c19740370692e87e9572973cfc40"],["E:/GitHubBlog/public/2020/04/02/leetcode记录-52/index.html","ffbe325a8fbdcfd4140455143e762894"],["E:/GitHubBlog/public/2020/04/17/Android第八周作业-每天第一次打开应用更新数据/index.html","20ee379dea91b9e3a23d63a0076c25f1"],["E:/GitHubBlog/public/2020/05/21/attention模型/index.html","3c12091aaeb9cbcef7685be463249670"],["E:/GitHubBlog/public/2020/05/21/编码器与解码器框架/index.html","2193f9b99a0fafd356627d110d84a507"],["E:/GitHubBlog/public/2020/07/11/思考，快与慢/index.html","0e07c3921203fe8bfd0c5c73f074cabc"],["E:/GitHubBlog/public/2020/07/13/20200713-0717总结/index.html","322d0b8268a5ba169280a1ab67735e3e"],["E:/GitHubBlog/public/2020/07/13/ZeroShotCeres Zero-Shot Relation Extraction/index.html","e4222f2c7e0faee09f722f514baf4178"],["E:/GitHubBlog/public/2020/07/20/Keyphrase Generation A Text Summarization Struggle/index.html","9c741c45cfb8314a3b5951378689a205"],["E:/GitHubBlog/public/2020/07/20/TPR/index.html","37ea694e0b0df3c3958198d0968d9469"],["E:/GitHubBlog/public/2020/07/23/Keyphrase Generation for Scientific Document Retrieval/index.html","eeea10e5715875c5828a182c98c827c5"],["E:/GitHubBlog/public/2020/07/24/20200718-0724总结/index.html","0af22de224055e88c73ae0f92594ab5a"],["E:/GitHubBlog/public/2020/07/31/20200725-0731总结/index.html","cc9d433657457901c855dfed668dbd8b"],["E:/GitHubBlog/public/2020/08/07/20200801-0807总结/index.html","b9a5be325731e10a54151fbbb889c280"],["E:/GitHubBlog/public/2020/08/12/逻辑回归算法/index.html","f4c57439baa3305b599bfbb2485544dd"],["E:/GitHubBlog/public/2020/08/13/李宏毅-机器学习21/index.html","f496b31d4cdf44695142812d613cd3b1"],["E:/GitHubBlog/public/2020/08/14/20200808-0815总结/index.html","f19d61187088ca95b993c35c69a5891f"],["E:/GitHubBlog/public/2020/09/03/20200816-0903总结/index.html","86574c2c3459852d030ef3e7641dd5c1"],["E:/GitHubBlog/public/2020/09/16/20200914-0918总结/index.html","de3fb4ad5213991465b52d13ad238621"],["E:/GitHubBlog/public/2020/09/25/20200919-0925总结/index.html","945f120d54b730bbb0992e28a8ed143a"],["E:/GitHubBlog/public/2020/10/06/《Deep Keyphrase Generation》详细解析！/index.html","884a91d3987758be5effd31b8ec53a32"],["E:/GitHubBlog/public/2020/10/16/20201008-1017总结/index.html","5a9d751c8d44ad6ad1ea0801c1b4389a"],["E:/GitHubBlog/public/2020/10/20/特征选择的几种方法(一)————Filter过滤法/index.html","2e31fd88ca818ff158f6cce86ba45276"],["E:/GitHubBlog/public/2020/10/21/PCA/index.html","325b854d3a7d5374d77cdbc1634945af"],["E:/GitHubBlog/public/2020/10/22/20201018-1023总结/index.html","16ed12997f06bfd97c1240d9aeba3f69"],["E:/GitHubBlog/public/2020/10/24/20201024-1030/index.html","e6f045f6415f28a7031975a833db5aa1"],["E:/GitHubBlog/public/2020/10/30/1030讨论记录/index.html","763d5cd5bbb9dd66071e335d281a0d02"],["E:/GitHubBlog/public/2020/11/02/《如何写一个商业计划书》读后感/index.html","20038c518e33d51618f451498ec9b7b3"],["E:/GitHubBlog/public/2020/11/03/20201103-总结/index.html","00953328f86e888cc9c558da7f32073d"],["E:/GitHubBlog/public/2020/11/04/《思考，快与慢》读后感/index.html","bcf65921905c10a3d14fa44788a77625"],["E:/GitHubBlog/public/archives/2020/01/index.html","4df72af1897e1736d27bd3dcab10cd82"],["E:/GitHubBlog/public/archives/2020/02/index.html","beafc027196cb408e20a14bd1d7e513d"],["E:/GitHubBlog/public/archives/2020/03/index.html","1d8fd139a40e7f96136eeea569e6127e"],["E:/GitHubBlog/public/archives/2020/04/index.html","bce52b715e24d14af4c3db3e0c7ac25d"],["E:/GitHubBlog/public/archives/2020/05/index.html","fd0bc6d4fd225e22856b0b72f2117cf5"],["E:/GitHubBlog/public/archives/2020/07/index.html","17ea232aebd0e87b2b303a4c63dca39f"],["E:/GitHubBlog/public/archives/2020/08/index.html","295247a9584c68933ddead71e4204136"],["E:/GitHubBlog/public/archives/2020/09/index.html","476ba19841b3ccc5a407761e0c162d99"],["E:/GitHubBlog/public/archives/2020/10/index.html","5ea70d36f78e937b6bcd580e416186d6"],["E:/GitHubBlog/public/archives/2020/11/index.html","68cdbe58d74b6852e2cb53e3e7a9f729"],["E:/GitHubBlog/public/archives/2020/index.html","024e4273dbff13e34297170335843345"],["E:/GitHubBlog/public/archives/2020/page/2/index.html","0d6830938581e675f028fe2a6ebe30ad"],["E:/GitHubBlog/public/archives/2020/page/3/index.html","b06aff597ed652377fd3cf3236620564"],["E:/GitHubBlog/public/archives/2020/page/4/index.html","88632f77cef18b37d19993ea7596bcab"],["E:/GitHubBlog/public/archives/2020/page/5/index.html","9e5a25ad98d3d220e121cf508107829b"],["E:/GitHubBlog/public/archives/index.html","2bb5e692e3e598dea0e8f73e5e24141c"],["E:/GitHubBlog/public/archives/page/2/index.html","bf0e10f78846e0f176120f9bcb6a3d5c"],["E:/GitHubBlog/public/archives/page/3/index.html","735f31205f6de046d5111b727a801465"],["E:/GitHubBlog/public/archives/page/4/index.html","484a3315778f8570230ce514029bc3ae"],["E:/GitHubBlog/public/archives/page/5/index.html","3b34216e23c47e9c08a1e01970267386"],["E:/GitHubBlog/public/css/404.css","b1bb50e1fabe41adcec483f6427fb3aa"],["E:/GitHubBlog/public/css/index.css","2389364df340948a5b822422956c2c87"],["E:/GitHubBlog/public/css/var.css","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/img/algolia.svg","fd40b88ac5370a5353a50b8175c1f367"],["E:/GitHubBlog/public/img/avatar.png","6cc4a809d23e3d8946a299ae4ce4e4cd"],["E:/GitHubBlog/public/index.html","a9a2c78de9e9962285cc85d27c49545a"],["E:/GitHubBlog/public/js/copy.js","45aae54bf2e43ac27ecc41eb5e0bacf7"],["E:/GitHubBlog/public/js/fancybox.js","f84d626654b9bbc05720952b3effe062"],["E:/GitHubBlog/public/js/fireworks.js","35933ce61c158ef9c33b5e089fe757ac"],["E:/GitHubBlog/public/js/head.js","347edd99f8e3921b45fa617e839d8182"],["E:/GitHubBlog/public/js/hexo-theme-melody.js","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/js/katex.js","d971ba8b8dee1120ef66744b89cfd2b1"],["E:/GitHubBlog/public/js/scroll.js","a12ad7e37b9e325f8da3d0523d3757c7"],["E:/GitHubBlog/public/js/search/algolia.js","53160985d32d6fd66d98aa0e05b081ac"],["E:/GitHubBlog/public/js/search/local-search.js","1565b508bd866ed6fbd724a3858af5d8"],["E:/GitHubBlog/public/js/sidebar.js","d24db780974e661198eeb2e45f20a28f"],["E:/GitHubBlog/public/js/third-party/anime.min.js","9b4bbe6deb700e1c3606eab732f5eea5"],["E:/GitHubBlog/public/js/third-party/canvas-ribbon.js","09c181544ddff1db701db02ac30453e0"],["E:/GitHubBlog/public/js/third-party/jquery.fancybox.min.js","3c9fa1c1199cd4f874d855ecb1641335"],["E:/GitHubBlog/public/js/third-party/jquery.min.js","c9f5aeeca3ad37bf2aa006139b935f0a"],["E:/GitHubBlog/public/js/third-party/reveal/head.min.js","aad121203010122e05f1766d92385214"],["E:/GitHubBlog/public/js/third-party/reveal/highlight.min.js","44594243bec43813a16371af8fe7e105"],["E:/GitHubBlog/public/js/third-party/reveal/markdown.min.js","7ec4cef5a7fe3f0bf0eb4dc6d7bca114"],["E:/GitHubBlog/public/js/third-party/reveal/marked.min.js","c2a88705e206d71dc21fdc4445349127"],["E:/GitHubBlog/public/js/third-party/reveal/math.min.js","0a278fee2e57c530ab07f7d2d9ea8d96"],["E:/GitHubBlog/public/js/third-party/reveal/notes.min.js","89a0dfae4d706f9c75b317f686c3aa14"],["E:/GitHubBlog/public/js/third-party/reveal/reveal.min.js","8988419d67efb5fe93e291a357e26ec9"],["E:/GitHubBlog/public/js/third-party/reveal/zoom.min.js","9791f96e63e7d534cba2b67d4bda0419"],["E:/GitHubBlog/public/js/third-party/velocity.min.js","64da069aba987ea0512cf610600a56d1"],["E:/GitHubBlog/public/js/third-party/velocity.ui.min.js","c8ca438424a080620f7b2f4ee4b0fff1"],["E:/GitHubBlog/public/js/transition.js","911db4268f0f6621073afcced9e1bfef"],["E:/GitHubBlog/public/js/utils.js","3ff3423d966a1c351e9867813b3f6d36"],["E:/GitHubBlog/public/page/2/index.html","f7b83184173f2e233cf921dcb03fa3d6"],["E:/GitHubBlog/public/page/3/index.html","b3a80520a601b4047bfc2376783c4327"],["E:/GitHubBlog/public/page/4/index.html","29bb8fe0a6f88a6df779f3a8884e7b64"],["E:/GitHubBlog/public/page/5/index.html","7d55df0dcf8addd6c9aa1486d61e2e49"],["E:/GitHubBlog/public/tags/Android/index.html","21d18c70af38d12edaf1d38116d4d30a"],["E:/GitHubBlog/public/tags/NLP/index.html","cbcfb187abd45eb1894084043918bbbf"],["E:/GitHubBlog/public/tags/NLP/page/2/index.html","8656f2d2181bda5082ab335d7a72035a"],["E:/GitHubBlog/public/tags/R/index.html","2b8099fe62a01bb134175ed01c45e245"],["E:/GitHubBlog/public/tags/index.html","fbb92457e84ace5a8ef006597b71747b"],["E:/GitHubBlog/public/tags/java/index.html","c4a47f149d59882164fe0a7abe319393"],["E:/GitHubBlog/public/tags/leetcode/index.html","543361a9afeb253084c557e14a45b8c4"],["E:/GitHubBlog/public/tags/python/index.html","7d61c6a723fa7ef3e7d6dc36c1be2100"],["E:/GitHubBlog/public/tags/总结/index.html","169439a0900973bc7564798c4465d860"],["E:/GitHubBlog/public/tags/总结/page/2/index.html","d838c113341604c741e53249a6fb8522"],["E:/GitHubBlog/public/tags/数据分析/index.html","0baf857a3e6bad4d1914b3eeff832e0a"],["E:/GitHubBlog/public/tags/数据挖掘/index.html","6a9ac4a7ef19611f1e8fcd3308c4ff51"],["E:/GitHubBlog/public/tags/数据结构/index.html","27bbba66e49c4036c2445ddd60ab57d7"],["E:/GitHubBlog/public/tags/机器学习/index.html","ba4ee2e7c0e021ff248741d04e2b143c"],["E:/GitHubBlog/public/tags/深度学习/index.html","da1bebc5e8567bf1b188ee8737fa9e6f"],["E:/GitHubBlog/public/tags/爬虫/index.html","938a2ee18ac3370637938ccebcd7cc8b"],["E:/GitHubBlog/public/tags/笔记/index.html","e5cb8d8b13eaff1b84f313cb39a89e83"],["E:/GitHubBlog/public/tags/论文/index.html","47e18d8659c8467bcb5e06febc054a69"],["E:/GitHubBlog/public/tags/论文/page/2/index.html","920b8398c833dd6b2c02a69bf504a5cc"],["E:/GitHubBlog/public/tags/读书笔记/index.html","6d0f93dc59e6a256e60a01a9fba2f49d"]];
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







