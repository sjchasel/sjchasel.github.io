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

var precacheConfig = [["E:/GitHubBlog/public/2020/01/28/hello-world/index.html","327fab29b14a593d7f729d606646a979"],["E:/GitHubBlog/public/2020/02/08/python爬虫学习总结/index.html","d950280711e1e1c2e8c7bf2e01be972b"],["E:/GitHubBlog/public/2020/02/19/数据结构第一周-绪论/index.html","74a53254f9f7f7a0be8bddb42ac62aaa"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-1/index.html","79baaed2a5500f3c1af4b3c54c3afd57"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-2-创建数据集/index.html","b680ee03004fe0e972156c3363664864"],["E:/GitHubBlog/public/2020/03/01/《R语言实战》学习总结-3-图形初阶/index.html","7c1cc9fed70b739f2f2a9e5adf5a5d6d"],["E:/GitHubBlog/public/2020/03/02/数据结构第二周-线性结构1/index.html","7e4ad65ea9685155d367f1880ee05396"],["E:/GitHubBlog/public/2020/03/03/《R语言实战》学习总结-4-基本数据管理/index.html","e9d688c1241fb06da9941bb6a0ef1118"],["E:/GitHubBlog/public/2020/03/14/数据结构第三周-线性结构2/index.html","b63fdaf6c71e7de140ebe3087f66003a"],["E:/GitHubBlog/public/2020/03/14/数据结构第四周-栈与队列/index.html","1e86bc345499a510523ee6b455fe30a2"],["E:/GitHubBlog/public/2020/03/17/leetcode记录-1、88、20/index.html","e3265333a1d167b12790ffc826009c5e"],["E:/GitHubBlog/public/2020/03/20/温度转换程序/index.html","58f3b84d2107382037f02774ea14325b"],["E:/GitHubBlog/public/2020/03/22/数据结构第五周-递归/index.html","29d04b5d9c4c7815b69a5620e513b034"],["E:/GitHubBlog/public/2020/03/22/数据结构第六周-串与数组/index.html","75a2144bf9be842913d4739af59b13f5"],["E:/GitHubBlog/public/2020/04/02/leetcode记录-52/index.html","de23e0f562804c513c941271d333f7dd"],["E:/GitHubBlog/public/2020/04/17/Android第八周作业-每天第一次打开应用更新数据/index.html","fe9e213e89d628b41410a65492d99fb8"],["E:/GitHubBlog/public/2020/05/21/attention模型/index.html","48960122492d87b01a7f6cd5d681e80b"],["E:/GitHubBlog/public/2020/05/21/编码器与解码器框架/index.html","44a51dbced756ab49ba7540fbcc9f0be"],["E:/GitHubBlog/public/2020/07/11/思考，快与慢/index.html","3b3c8e06c8ff6ccc1e5e6b9dcc8d3b64"],["E:/GitHubBlog/public/2020/07/13/20200713-0717总结/index.html","5308587c6d01a4b7a039c26706a13005"],["E:/GitHubBlog/public/2020/07/13/ZeroShotCeres Zero-Shot Relation Extraction/index.html","b76e84ed01f290dbd6035524423dfb10"],["E:/GitHubBlog/public/2020/07/13/《lingo与excel在数学建模中的应用》/index.html","98df043dd16f692d575a74c05af7517b"],["E:/GitHubBlog/public/2020/07/20/Keyphrase Generation A Text Summarization Struggle/index.html","e78117cf67dce89e802e91a1a47bac5a"],["E:/GitHubBlog/public/2020/07/20/TPR/index.html","ec8e2007de8cabd891b646ef4b4185f3"],["E:/GitHubBlog/public/2020/07/21/天池NLP赛事-新闻文本分类-Task1-赛题理解/index.html","74f83dad430d54f3daf8bd315f01990b"],["E:/GitHubBlog/public/2020/07/22/天池NLP赛事-新闻文本分类-Task2-数据读取与数据分析/index.html","ad2650075a2aadd3a8ff5e8da11be682"],["E:/GitHubBlog/public/2020/07/23/Keyphrase Generation for Scientific Document Retrieval/index.html","25c79adb94471188e38da472f4b71dd2"],["E:/GitHubBlog/public/2020/07/24/20200718-0724总结/index.html","685f86be86f43c31b29e670b9a793d1d"],["E:/GitHubBlog/public/2020/07/25/天池NLP赛事-新闻文本分类-Task3-基于机器学习的文本分类/index.html","1d7764ae27f44f0d5dbd92280b334efc"],["E:/GitHubBlog/public/2020/07/27/基于深度学习的文本分类/index.html","942405c5999acb78ec13170e5fc645a9"],["E:/GitHubBlog/public/archives/2020/01/index.html","681d8b4436fb1b0bd7ca153e835ac971"],["E:/GitHubBlog/public/archives/2020/02/index.html","80128c6775389e038d2000997f78256a"],["E:/GitHubBlog/public/archives/2020/03/index.html","1be8433abec919abe82e85e16503ecd6"],["E:/GitHubBlog/public/archives/2020/04/index.html","3f7c7be7ce52112e3da0f772211487f7"],["E:/GitHubBlog/public/archives/2020/05/index.html","34e8b305413434610b7f179033b14dcf"],["E:/GitHubBlog/public/archives/2020/07/index.html","d972415bb49c64dabd987748bc2dddc2"],["E:/GitHubBlog/public/archives/2020/07/page/2/index.html","5695e53a12634d26a17c46a3257f6512"],["E:/GitHubBlog/public/archives/2020/index.html","36541965deda2a2724aca05f6b713eaf"],["E:/GitHubBlog/public/archives/2020/page/2/index.html","fec62f7f58bd1586ff3880fb37a0fda1"],["E:/GitHubBlog/public/archives/2020/page/3/index.html","547bf9f9d9fd828a7d1dda100245da57"],["E:/GitHubBlog/public/archives/index.html","f89c69037e2840f51e1e2ddeb95554fb"],["E:/GitHubBlog/public/archives/page/2/index.html","e455de59cd588bb402b7479987d3c871"],["E:/GitHubBlog/public/archives/page/3/index.html","5a874a0daa9a23b1f46f92e3f48266b1"],["E:/GitHubBlog/public/css/404.css","b1bb50e1fabe41adcec483f6427fb3aa"],["E:/GitHubBlog/public/css/index.css","2389364df340948a5b822422956c2c87"],["E:/GitHubBlog/public/css/var.css","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/img/algolia.svg","fd40b88ac5370a5353a50b8175c1f367"],["E:/GitHubBlog/public/img/avatar.png","6cc4a809d23e3d8946a299ae4ce4e4cd"],["E:/GitHubBlog/public/index.html","4c0d0e46bd9770867d0d7ff61bc216c9"],["E:/GitHubBlog/public/js/copy.js","45aae54bf2e43ac27ecc41eb5e0bacf7"],["E:/GitHubBlog/public/js/fancybox.js","f84d626654b9bbc05720952b3effe062"],["E:/GitHubBlog/public/js/fireworks.js","35933ce61c158ef9c33b5e089fe757ac"],["E:/GitHubBlog/public/js/head.js","347edd99f8e3921b45fa617e839d8182"],["E:/GitHubBlog/public/js/hexo-theme-melody.js","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/js/katex.js","d971ba8b8dee1120ef66744b89cfd2b1"],["E:/GitHubBlog/public/js/scroll.js","a12ad7e37b9e325f8da3d0523d3757c7"],["E:/GitHubBlog/public/js/search/algolia.js","53160985d32d6fd66d98aa0e05b081ac"],["E:/GitHubBlog/public/js/search/local-search.js","1565b508bd866ed6fbd724a3858af5d8"],["E:/GitHubBlog/public/js/sidebar.js","d24db780974e661198eeb2e45f20a28f"],["E:/GitHubBlog/public/js/third-party/anime.min.js","9b4bbe6deb700e1c3606eab732f5eea5"],["E:/GitHubBlog/public/js/third-party/canvas-ribbon.js","09c181544ddff1db701db02ac30453e0"],["E:/GitHubBlog/public/js/third-party/jquery.fancybox.min.js","3c9fa1c1199cd4f874d855ecb1641335"],["E:/GitHubBlog/public/js/third-party/jquery.min.js","c9f5aeeca3ad37bf2aa006139b935f0a"],["E:/GitHubBlog/public/js/third-party/reveal/head.min.js","aad121203010122e05f1766d92385214"],["E:/GitHubBlog/public/js/third-party/reveal/highlight.min.js","44594243bec43813a16371af8fe7e105"],["E:/GitHubBlog/public/js/third-party/reveal/markdown.min.js","7ec4cef5a7fe3f0bf0eb4dc6d7bca114"],["E:/GitHubBlog/public/js/third-party/reveal/marked.min.js","c2a88705e206d71dc21fdc4445349127"],["E:/GitHubBlog/public/js/third-party/reveal/math.min.js","0a278fee2e57c530ab07f7d2d9ea8d96"],["E:/GitHubBlog/public/js/third-party/reveal/notes.min.js","89a0dfae4d706f9c75b317f686c3aa14"],["E:/GitHubBlog/public/js/third-party/reveal/reveal.min.js","8988419d67efb5fe93e291a357e26ec9"],["E:/GitHubBlog/public/js/third-party/reveal/zoom.min.js","9791f96e63e7d534cba2b67d4bda0419"],["E:/GitHubBlog/public/js/third-party/velocity.min.js","64da069aba987ea0512cf610600a56d1"],["E:/GitHubBlog/public/js/third-party/velocity.ui.min.js","c8ca438424a080620f7b2f4ee4b0fff1"],["E:/GitHubBlog/public/js/transition.js","911db4268f0f6621073afcced9e1bfef"],["E:/GitHubBlog/public/js/utils.js","3ff3423d966a1c351e9867813b3f6d36"],["E:/GitHubBlog/public/page/2/index.html","d1a8a86e00878e7e2348678a4c821275"],["E:/GitHubBlog/public/page/3/index.html","bb10a3315912fbc5b922761ac1e7e2df"],["E:/GitHubBlog/public/tags/Android/index.html","5ac9d115ba1c0c9afa2d573df37928c7"],["E:/GitHubBlog/public/tags/NLP/index.html","ad64af8b77a816cd6fcd7fbee6cec837"],["E:/GitHubBlog/public/tags/R/index.html","53bb26b57e86885da647b8550255d173"],["E:/GitHubBlog/public/tags/index.html","ae95328a288729de787e6f1eefca9277"],["E:/GitHubBlog/public/tags/java/index.html","7ce4f05c782f5aa0b81a0d9688d66dfa"],["E:/GitHubBlog/public/tags/leetcode/index.html","eef06974f5e7af31a1962b62d6d92ea0"],["E:/GitHubBlog/public/tags/lingo/index.html","f0cf49fc1dfe7399a2172b13b453eee6"],["E:/GitHubBlog/public/tags/python/index.html","4469ab15444a78c28001a1ac3c26dcf9"],["E:/GitHubBlog/public/tags/总结/index.html","a731c7ae66d0ace4dae8d49751a60e1d"],["E:/GitHubBlog/public/tags/数据分析/index.html","3a6f5385599bc53c55887143c63f5c6e"],["E:/GitHubBlog/public/tags/数据结构/index.html","5c808a4b92bff41dd12d0272c57d84a0"],["E:/GitHubBlog/public/tags/数模/index.html","69d17bc0e60ff3d1bdac9a1234bdbc7e"],["E:/GitHubBlog/public/tags/比赛/index.html","6e129cdae34576a626cca91bfb4bd819"],["E:/GitHubBlog/public/tags/深度学习/index.html","c09bd1c792305b3758737a105aa0368d"],["E:/GitHubBlog/public/tags/爬虫/index.html","19b33fd8236d40c561772ef1bebae475"],["E:/GitHubBlog/public/tags/论文/index.html","3b50425255fe3b230737e2002edb8b15"],["E:/GitHubBlog/public/tags/读书笔记/index.html","c86ffa15c23c126aefd45ac16fba30fb"]];
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







