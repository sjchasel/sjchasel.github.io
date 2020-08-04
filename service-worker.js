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

var precacheConfig = [["E:/GitHubBlog/public/2020/01/28/hello-world/index.html","58c5a5aa55d5bd4a08ddb87c08a22931"],["E:/GitHubBlog/public/2020/02/08/python爬虫学习总结/index.html","446cbd6e34b5900beee01521e42ba1c9"],["E:/GitHubBlog/public/2020/02/19/数据结构第一周-绪论/index.html","fefc34cd8473878638aa8467982fa3c2"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-1/index.html","371dedca9c9a3e14283f6d98a5a2ae05"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-2-创建数据集/index.html","706eb8831f914b3fc19f89b653c49575"],["E:/GitHubBlog/public/2020/03/01/《R语言实战》学习总结-3-图形初阶/index.html","adf757f58082e5c807442158cba7cb7d"],["E:/GitHubBlog/public/2020/03/02/数据结构第二周-线性结构1/index.html","eda660763e2e93343b00404fb1a28cb8"],["E:/GitHubBlog/public/2020/03/03/《R语言实战》学习总结-4-基本数据管理/index.html","95cfab1ea31d1ce03d8fc375d1c32bef"],["E:/GitHubBlog/public/2020/03/14/数据结构第三周-线性结构2/index.html","3416d1968171fb8b1a229811446fa828"],["E:/GitHubBlog/public/2020/03/14/数据结构第四周-栈与队列/index.html","57cb55eb3d74461efc750cacc1466547"],["E:/GitHubBlog/public/2020/03/17/leetcode记录-1、88、20/index.html","5f371a3864e5ec6ad3f192419bf85044"],["E:/GitHubBlog/public/2020/03/20/温度转换程序/index.html","5cd7a982c35c21f48614c9d17ec5b1ba"],["E:/GitHubBlog/public/2020/03/22/数据结构第五周-递归/index.html","f75fbbd730736b3ecd1e6e6f6bbc3cff"],["E:/GitHubBlog/public/2020/03/22/数据结构第六周-串与数组/index.html","4d9fc9ac9bf5768540da7602bc5538bd"],["E:/GitHubBlog/public/2020/04/02/leetcode记录-52/index.html","3e9541400f3fccf26162c5cca5e7c524"],["E:/GitHubBlog/public/2020/04/17/Android第八周作业-每天第一次打开应用更新数据/index.html","c7086777f073ef5f8099242989161e01"],["E:/GitHubBlog/public/2020/05/21/attention模型/index.html","16395a4d0729010291dc3c024c6c8cce"],["E:/GitHubBlog/public/2020/05/21/编码器与解码器框架/index.html","d88001f0242290db135b364bb669d9ec"],["E:/GitHubBlog/public/2020/07/11/思考，快与慢/index.html","fe21a152dfd480b8af84a86279bac4fd"],["E:/GitHubBlog/public/2020/07/13/20200713-0717总结/index.html","09cd6cf9ce7bea98aa183802e79e178d"],["E:/GitHubBlog/public/2020/07/13/ZeroShotCeres Zero-Shot Relation Extraction/index.html","a48f66f9092f358470ad70569582535d"],["E:/GitHubBlog/public/2020/07/13/《lingo与excel在数学建模中的应用》/index.html","6e2a1ea8c52a64967a511c2871f1022a"],["E:/GitHubBlog/public/2020/07/20/Keyphrase Generation A Text Summarization Struggle/index.html","94ba4a20fb65a455d121535e8bbc8e72"],["E:/GitHubBlog/public/2020/07/20/TPR/index.html","4d9c1da5bd78e99c155fe8dd4f52dc03"],["E:/GitHubBlog/public/2020/07/21/天池NLP赛事-新闻文本分类-Task1-赛题理解/index.html","a4cd9602c956508d62f54b180f234ad3"],["E:/GitHubBlog/public/2020/07/22/天池NLP赛事-新闻文本分类-Task2-数据读取与数据分析/index.html","7d16c55e22436cef042d728e0622de2d"],["E:/GitHubBlog/public/2020/07/23/Keyphrase Generation for Scientific Document Retrieval/index.html","3f9a82b5e8b22314a3c103c102839bea"],["E:/GitHubBlog/public/2020/07/24/20200718-0724总结/index.html","2f9fee8c0a5d068b1f13ab4a57d80c31"],["E:/GitHubBlog/public/2020/07/25/天池NLP赛事-新闻文本分类-Task3-基于机器学习的文本分类/index.html","5cc6b90218b7c11332eb73caf06bbec5"],["E:/GitHubBlog/public/2020/07/27/基于深度学习的文本分类/index.html","527141d6f0729e33dd3e1705c92eda16"],["E:/GitHubBlog/public/2020/07/31/20200725-0731总结/index.html","94dee8fde61113d21e6e69abac64066a"],["E:/GitHubBlog/public/2020/07/31/Task5/index.html","950c8968e7eb05861b32d1d7366c361c"],["E:/GitHubBlog/public/2020/08/04/基于深度学习的文本分类3/index.html","5d4052653c5e2e740205e96a1338487a"],["E:/GitHubBlog/public/archives/2020/01/index.html","53902bf070c06ea912fdf534ae227ab4"],["E:/GitHubBlog/public/archives/2020/02/index.html","54ad5acf5d75e680fc6f005a3950ba95"],["E:/GitHubBlog/public/archives/2020/03/index.html","4627fdfcd31c60f4a510cd6c96637023"],["E:/GitHubBlog/public/archives/2020/04/index.html","1a8b53f2589382b0ebd5b12de2b5d544"],["E:/GitHubBlog/public/archives/2020/05/index.html","9e3031fcfa84462e519f25b5879d804b"],["E:/GitHubBlog/public/archives/2020/07/index.html","7b2728c4da11d4374e3221485c9607d8"],["E:/GitHubBlog/public/archives/2020/07/page/2/index.html","966eef5e39d0b50da3232e2dca42f952"],["E:/GitHubBlog/public/archives/2020/08/index.html","f073b5b96f3b3811bf1e9baaf29ec25e"],["E:/GitHubBlog/public/archives/2020/index.html","02fe28ed0034b019da23a41c61a28d03"],["E:/GitHubBlog/public/archives/2020/page/2/index.html","3bc3b486ce1e9a8d1a945c5fe56b3e38"],["E:/GitHubBlog/public/archives/2020/page/3/index.html","ce573bbf40efb596c190cd342168b25a"],["E:/GitHubBlog/public/archives/2020/page/4/index.html","142aa0b6a26629f54e3292e813d6f87f"],["E:/GitHubBlog/public/archives/index.html","40c98fb25766b360e5b8360d56458d16"],["E:/GitHubBlog/public/archives/page/2/index.html","7e241b875c5b4e2ceb08c2b68f375b28"],["E:/GitHubBlog/public/archives/page/3/index.html","d0948cb3b5b2f0dcae544d08defc3aa0"],["E:/GitHubBlog/public/archives/page/4/index.html","ca2e5b4fe1e8d3a2fa27df866aba180c"],["E:/GitHubBlog/public/css/404.css","b1bb50e1fabe41adcec483f6427fb3aa"],["E:/GitHubBlog/public/css/index.css","2389364df340948a5b822422956c2c87"],["E:/GitHubBlog/public/css/var.css","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/img/algolia.svg","fd40b88ac5370a5353a50b8175c1f367"],["E:/GitHubBlog/public/img/avatar.png","6cc4a809d23e3d8946a299ae4ce4e4cd"],["E:/GitHubBlog/public/index.html","014b3eb3f52e9b2c6c333baa1fb26865"],["E:/GitHubBlog/public/js/copy.js","45aae54bf2e43ac27ecc41eb5e0bacf7"],["E:/GitHubBlog/public/js/fancybox.js","f84d626654b9bbc05720952b3effe062"],["E:/GitHubBlog/public/js/fireworks.js","35933ce61c158ef9c33b5e089fe757ac"],["E:/GitHubBlog/public/js/head.js","347edd99f8e3921b45fa617e839d8182"],["E:/GitHubBlog/public/js/hexo-theme-melody.js","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/js/katex.js","d971ba8b8dee1120ef66744b89cfd2b1"],["E:/GitHubBlog/public/js/scroll.js","a12ad7e37b9e325f8da3d0523d3757c7"],["E:/GitHubBlog/public/js/search/algolia.js","53160985d32d6fd66d98aa0e05b081ac"],["E:/GitHubBlog/public/js/search/local-search.js","1565b508bd866ed6fbd724a3858af5d8"],["E:/GitHubBlog/public/js/sidebar.js","d24db780974e661198eeb2e45f20a28f"],["E:/GitHubBlog/public/js/third-party/anime.min.js","9b4bbe6deb700e1c3606eab732f5eea5"],["E:/GitHubBlog/public/js/third-party/canvas-ribbon.js","09c181544ddff1db701db02ac30453e0"],["E:/GitHubBlog/public/js/third-party/jquery.fancybox.min.js","3c9fa1c1199cd4f874d855ecb1641335"],["E:/GitHubBlog/public/js/third-party/jquery.min.js","c9f5aeeca3ad37bf2aa006139b935f0a"],["E:/GitHubBlog/public/js/third-party/reveal/head.min.js","aad121203010122e05f1766d92385214"],["E:/GitHubBlog/public/js/third-party/reveal/highlight.min.js","44594243bec43813a16371af8fe7e105"],["E:/GitHubBlog/public/js/third-party/reveal/markdown.min.js","7ec4cef5a7fe3f0bf0eb4dc6d7bca114"],["E:/GitHubBlog/public/js/third-party/reveal/marked.min.js","c2a88705e206d71dc21fdc4445349127"],["E:/GitHubBlog/public/js/third-party/reveal/math.min.js","0a278fee2e57c530ab07f7d2d9ea8d96"],["E:/GitHubBlog/public/js/third-party/reveal/notes.min.js","89a0dfae4d706f9c75b317f686c3aa14"],["E:/GitHubBlog/public/js/third-party/reveal/reveal.min.js","8988419d67efb5fe93e291a357e26ec9"],["E:/GitHubBlog/public/js/third-party/reveal/zoom.min.js","9791f96e63e7d534cba2b67d4bda0419"],["E:/GitHubBlog/public/js/third-party/velocity.min.js","64da069aba987ea0512cf610600a56d1"],["E:/GitHubBlog/public/js/third-party/velocity.ui.min.js","c8ca438424a080620f7b2f4ee4b0fff1"],["E:/GitHubBlog/public/js/transition.js","911db4268f0f6621073afcced9e1bfef"],["E:/GitHubBlog/public/js/utils.js","3ff3423d966a1c351e9867813b3f6d36"],["E:/GitHubBlog/public/page/2/index.html","f94caacfe54c59e688d8adf32fa5715f"],["E:/GitHubBlog/public/page/3/index.html","1126686b9788fb23e8ca243daf532b3c"],["E:/GitHubBlog/public/page/4/index.html","d43e84a227dfef2e0f0f79604b812f19"],["E:/GitHubBlog/public/tags/Android/index.html","38a3879e16d65983730a7c053412c74c"],["E:/GitHubBlog/public/tags/NLP/index.html","2a1ba00aeac1440489371b6a2578eb7d"],["E:/GitHubBlog/public/tags/NLP/page/2/index.html","185b2a1a06ccbe553a286baeb645b036"],["E:/GitHubBlog/public/tags/R/index.html","7ac31f0ac1a6387a9b6c90cda4fa4506"],["E:/GitHubBlog/public/tags/index.html","cb27723e24fe73cc77f78f9baafcf632"],["E:/GitHubBlog/public/tags/java/index.html","e910fd9b763cab9805eb88f5e77e1526"],["E:/GitHubBlog/public/tags/leetcode/index.html","e30a91bb277fda38261a32bfd341c851"],["E:/GitHubBlog/public/tags/lingo/index.html","74e9a0def9881ac85ca649c638b5b843"],["E:/GitHubBlog/public/tags/python/index.html","33f974b287905ac52916564002963c76"],["E:/GitHubBlog/public/tags/总结/index.html","a124cb19fd80285a184f5800157e32e3"],["E:/GitHubBlog/public/tags/数据分析/index.html","2df62c98953cca6674c99fef17111670"],["E:/GitHubBlog/public/tags/数据结构/index.html","12a090b6157b9143f5327568543874b8"],["E:/GitHubBlog/public/tags/数模/index.html","1f58148802142793e2e416cbc024b013"],["E:/GitHubBlog/public/tags/比赛/index.html","6b0323dc7ae278f4946ca342a98f9dcf"],["E:/GitHubBlog/public/tags/深度学习/index.html","f0a711660fdba8d7fd8c7be06917d129"],["E:/GitHubBlog/public/tags/爬虫/index.html","6b1ceb4a892e99093fd46951d497c2ea"],["E:/GitHubBlog/public/tags/论文/index.html","91f7f2b53bd1d33d26aee5a2ddb0f1c0"],["E:/GitHubBlog/public/tags/读书笔记/index.html","6be07c2c834a2577f76d9edb733d0fc8"]];
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







