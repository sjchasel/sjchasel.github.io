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

var precacheConfig = [["E:/GitHubBlog/public/2020/01/28/hello-world/index.html","26a71ffadb90030cc112c81382147ac5"],["E:/GitHubBlog/public/2020/02/08/python爬虫学习总结/index.html","deb7e9921d2014ad97ba54b2f209b945"],["E:/GitHubBlog/public/2020/02/19/数据结构第一周-绪论/index.html","57946ed7ad8672aa881b3ecd48092d5b"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-1/index.html","9abdb82e5e9b570154c55953f40a8e5e"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-2-创建数据集/index.html","d2079308988a9826c7c301568f148779"],["E:/GitHubBlog/public/2020/03/01/《R语言实战》学习总结-3-图形初阶/index.html","a65ec4551868ff984a4d22af3e6ee07d"],["E:/GitHubBlog/public/2020/03/02/数据结构第二周-线性结构1/index.html","56b3fed02862c8d5e4caa2d1f7db1302"],["E:/GitHubBlog/public/2020/03/03/《R语言实战》学习总结-4-基本数据管理/index.html","355d70fbbb5adfff688af6f81e8c1ac2"],["E:/GitHubBlog/public/2020/03/14/数据结构第三周-线性结构2/index.html","bf97ed1cdc45c1a24b62041c8900ead5"],["E:/GitHubBlog/public/2020/03/14/数据结构第四周-栈与队列/index.html","934a1a77bfaf0364fbc7d85580e2c0d1"],["E:/GitHubBlog/public/2020/03/17/leetcode记录-1、88、20/index.html","58bf75486967ff4987ec8af216518503"],["E:/GitHubBlog/public/2020/03/20/温度转换程序/index.html","5547c99302ff72ff47cabdf9555aceb8"],["E:/GitHubBlog/public/2020/03/22/数据结构第五周-递归/index.html","353e254e831206e9704c4e2d0679a4ca"],["E:/GitHubBlog/public/2020/03/22/数据结构第六周-串与数组/index.html","6104be3ffe41e9443bf8f2e914107cdc"],["E:/GitHubBlog/public/2020/04/02/leetcode记录-52/index.html","797d5fc2f4313a22ef8fd6235de19d06"],["E:/GitHubBlog/public/2020/04/17/Android第八周作业-每天第一次打开应用更新数据/index.html","78bf3294a3473ef717423dd2f88795aa"],["E:/GitHubBlog/public/2020/05/21/attention模型/index.html","aebd3fb5777a6888bf32a002cfd41567"],["E:/GitHubBlog/public/2020/05/21/编码器与解码器框架/index.html","284c82e4338e0ff4ba09d63ef9fa3d04"],["E:/GitHubBlog/public/2020/07/11/思考，快与慢/index.html","10f720c0b9f44df3a9f6a802b28f3a9b"],["E:/GitHubBlog/public/2020/07/13/20200713-0717总结/index.html","bebed8c2df3736db66a3b2f4966c8dd3"],["E:/GitHubBlog/public/2020/07/13/ZeroShotCeres Zero-Shot Relation Extraction/index.html","297e5c74b7bf6a7e50cac9dd76dc767e"],["E:/GitHubBlog/public/2020/07/13/《lingo与excel在数学建模中的应用》/index.html","b79c7579779c4a0d6b0f0a3698aaa786"],["E:/GitHubBlog/public/2020/07/20/Keyphrase Generation A Text Summarization Struggle/index.html","c374778b88553d7927361f5466ef9a16"],["E:/GitHubBlog/public/2020/07/20/TPR/index.html","cd606ce2cdeedd4138d9101686c85c0f"],["E:/GitHubBlog/public/2020/07/21/天池NLP赛事-新闻文本分类-Task1-赛题理解/index.html","a56ef4769f55273810b1096f822c0381"],["E:/GitHubBlog/public/2020/07/22/天池NLP赛事-新闻文本分类-Task2-数据读取与数据分析/index.html","27e5aa0e2ce1d1329d8a7b8efeb638dc"],["E:/GitHubBlog/public/2020/07/23/Keyphrase Generation for Scientific Document Retrieval/index.html","0c7c7260e97596f6b449264635f435b7"],["E:/GitHubBlog/public/2020/07/24/20200718-0724总结/index.html","f563ee954933b100844293fe17e9d6db"],["E:/GitHubBlog/public/2020/07/25/天池NLP赛事-新闻文本分类-Task3-基于机器学习的文本分类/index.html","d9c96ec6d7c0cfc2a2939c23930e4ef5"],["E:/GitHubBlog/public/archives/2020/01/index.html","149786aa01b1edb2b1c1602d217a25d0"],["E:/GitHubBlog/public/archives/2020/02/index.html","4d52d1b6b752644a6fab0cba0b540aa1"],["E:/GitHubBlog/public/archives/2020/03/index.html","b8d44c62d8cdcca4c1ff235fec9e45f2"],["E:/GitHubBlog/public/archives/2020/04/index.html","09ca9055bc2f9b8f7c874b8931ef00ef"],["E:/GitHubBlog/public/archives/2020/05/index.html","f54aefa6a376cfa2c116dc26684e2b03"],["E:/GitHubBlog/public/archives/2020/07/index.html","5412159e8e8a54926d760b791de29d45"],["E:/GitHubBlog/public/archives/2020/07/page/2/index.html","074351a03b619c0bd8f2726e9e42897c"],["E:/GitHubBlog/public/archives/2020/index.html","d0ddc3498b51b7de0f26515b9010f7cb"],["E:/GitHubBlog/public/archives/2020/page/2/index.html","abed984ff23a64d10f4629e8d8fa4f37"],["E:/GitHubBlog/public/archives/2020/page/3/index.html","7e3c91dba7b7d6169643a7d7a7c1d70b"],["E:/GitHubBlog/public/archives/index.html","dadda4bfdb4a1ab668619b01f7f31581"],["E:/GitHubBlog/public/archives/page/2/index.html","72a22b7058510859ca964dc8740c021d"],["E:/GitHubBlog/public/archives/page/3/index.html","ea1723a7c8d66e13264c8e1a0a64c567"],["E:/GitHubBlog/public/css/404.css","b1bb50e1fabe41adcec483f6427fb3aa"],["E:/GitHubBlog/public/css/index.css","2389364df340948a5b822422956c2c87"],["E:/GitHubBlog/public/css/var.css","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/img/algolia.svg","fd40b88ac5370a5353a50b8175c1f367"],["E:/GitHubBlog/public/img/avatar.png","6cc4a809d23e3d8946a299ae4ce4e4cd"],["E:/GitHubBlog/public/index.html","79778c706fab708c4fbee4f375336e0b"],["E:/GitHubBlog/public/js/copy.js","45aae54bf2e43ac27ecc41eb5e0bacf7"],["E:/GitHubBlog/public/js/fancybox.js","f84d626654b9bbc05720952b3effe062"],["E:/GitHubBlog/public/js/fireworks.js","35933ce61c158ef9c33b5e089fe757ac"],["E:/GitHubBlog/public/js/head.js","347edd99f8e3921b45fa617e839d8182"],["E:/GitHubBlog/public/js/hexo-theme-melody.js","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/js/katex.js","d971ba8b8dee1120ef66744b89cfd2b1"],["E:/GitHubBlog/public/js/scroll.js","a12ad7e37b9e325f8da3d0523d3757c7"],["E:/GitHubBlog/public/js/search/algolia.js","53160985d32d6fd66d98aa0e05b081ac"],["E:/GitHubBlog/public/js/search/local-search.js","1565b508bd866ed6fbd724a3858af5d8"],["E:/GitHubBlog/public/js/sidebar.js","d24db780974e661198eeb2e45f20a28f"],["E:/GitHubBlog/public/js/third-party/anime.min.js","9b4bbe6deb700e1c3606eab732f5eea5"],["E:/GitHubBlog/public/js/third-party/canvas-ribbon.js","09c181544ddff1db701db02ac30453e0"],["E:/GitHubBlog/public/js/third-party/jquery.fancybox.min.js","3c9fa1c1199cd4f874d855ecb1641335"],["E:/GitHubBlog/public/js/third-party/jquery.min.js","c9f5aeeca3ad37bf2aa006139b935f0a"],["E:/GitHubBlog/public/js/third-party/reveal/head.min.js","aad121203010122e05f1766d92385214"],["E:/GitHubBlog/public/js/third-party/reveal/highlight.min.js","44594243bec43813a16371af8fe7e105"],["E:/GitHubBlog/public/js/third-party/reveal/markdown.min.js","7ec4cef5a7fe3f0bf0eb4dc6d7bca114"],["E:/GitHubBlog/public/js/third-party/reveal/marked.min.js","c2a88705e206d71dc21fdc4445349127"],["E:/GitHubBlog/public/js/third-party/reveal/math.min.js","0a278fee2e57c530ab07f7d2d9ea8d96"],["E:/GitHubBlog/public/js/third-party/reveal/notes.min.js","89a0dfae4d706f9c75b317f686c3aa14"],["E:/GitHubBlog/public/js/third-party/reveal/reveal.min.js","8988419d67efb5fe93e291a357e26ec9"],["E:/GitHubBlog/public/js/third-party/reveal/zoom.min.js","9791f96e63e7d534cba2b67d4bda0419"],["E:/GitHubBlog/public/js/third-party/velocity.min.js","64da069aba987ea0512cf610600a56d1"],["E:/GitHubBlog/public/js/third-party/velocity.ui.min.js","c8ca438424a080620f7b2f4ee4b0fff1"],["E:/GitHubBlog/public/js/transition.js","911db4268f0f6621073afcced9e1bfef"],["E:/GitHubBlog/public/js/utils.js","3ff3423d966a1c351e9867813b3f6d36"],["E:/GitHubBlog/public/page/2/index.html","ecddb8fbfaf8eb982f035b0ab31273e7"],["E:/GitHubBlog/public/page/3/index.html","e7f45cd1e633c3babd95d10b6e6fd632"],["E:/GitHubBlog/public/tags/Android/index.html","fb20c4aea17f50e5d0c1f74b4a451685"],["E:/GitHubBlog/public/tags/NLP/index.html","3cda34ef31118a5025ec83b2962fce11"],["E:/GitHubBlog/public/tags/R/index.html","135aaa1f1722b5c742e6f8ff623114c9"],["E:/GitHubBlog/public/tags/index.html","c3b380899b85362530c788f88d652cb2"],["E:/GitHubBlog/public/tags/java/index.html","26289c827dd4db3f8eea6f3f1a1da61d"],["E:/GitHubBlog/public/tags/leetcode/index.html","c550d56bc9aa26572032d96e40e5219a"],["E:/GitHubBlog/public/tags/lingo/index.html","c9f1f99fd6897dde0af755739f6b455b"],["E:/GitHubBlog/public/tags/python/index.html","e247c481c9d179d3022d674037347af6"],["E:/GitHubBlog/public/tags/总结/index.html","88fdd837d148273691e9902bed946940"],["E:/GitHubBlog/public/tags/数据分析/index.html","798be1f05aa3c9adbd029772f8c37b30"],["E:/GitHubBlog/public/tags/数据结构/index.html","99ff6bbb8ba0db48752f5bf4c9aff787"],["E:/GitHubBlog/public/tags/数模/index.html","99d6f3bdb6ec4a21b6ea1bb5f731d408"],["E:/GitHubBlog/public/tags/比赛/index.html","add848cda71a9e38d198dbe639fc7ac4"],["E:/GitHubBlog/public/tags/深度学习/index.html","e48e4440029f1e263852802743a34f63"],["E:/GitHubBlog/public/tags/爬虫/index.html","b62aac6b6f7a7555c090d08ef4cc2eb5"],["E:/GitHubBlog/public/tags/论文/index.html","a35558548ecbb30dd1f9f8291bfc1073"],["E:/GitHubBlog/public/tags/读书笔记/index.html","09a44fe2281f3ed5fd0cab9c640e0cbe"]];
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







