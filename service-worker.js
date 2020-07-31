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

var precacheConfig = [["E:/GitHubBlog/public/2020/01/28/hello-world/index.html","8fa17abc0deb9a2e62d508114d68afd5"],["E:/GitHubBlog/public/2020/02/08/python爬虫学习总结/index.html","8092a58f5862fb0adafe534cffa323f6"],["E:/GitHubBlog/public/2020/02/19/数据结构第一周-绪论/index.html","17931b398c6cd840ae99e7eb7467b024"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-1/index.html","d9b125da94131aff4873b62d424f55e9"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-2-创建数据集/index.html","3cb6545a43abf36c4376d0f38815a95d"],["E:/GitHubBlog/public/2020/03/01/《R语言实战》学习总结-3-图形初阶/index.html","0133a80deab9952207102063062a81b2"],["E:/GitHubBlog/public/2020/03/02/数据结构第二周-线性结构1/index.html","bc3b22183c4425f0c5252df563aa5d87"],["E:/GitHubBlog/public/2020/03/03/《R语言实战》学习总结-4-基本数据管理/index.html","c471da762802210d3d04ea007274bef4"],["E:/GitHubBlog/public/2020/03/14/数据结构第三周-线性结构2/index.html","cc7099cc5db1a00d45404e33a1873233"],["E:/GitHubBlog/public/2020/03/14/数据结构第四周-栈与队列/index.html","926f2d1f90421b4adc9009cae7f6dbd1"],["E:/GitHubBlog/public/2020/03/17/leetcode记录-1、88、20/index.html","9b8994e84b46a726f4d02a954873d738"],["E:/GitHubBlog/public/2020/03/20/温度转换程序/index.html","56ba97906f0a915c9952363e16cf1fb3"],["E:/GitHubBlog/public/2020/03/22/数据结构第五周-递归/index.html","125d6c36ed1b6fd7b405b3b935072768"],["E:/GitHubBlog/public/2020/03/22/数据结构第六周-串与数组/index.html","8add83a37b2532bc147a7e7f47083f23"],["E:/GitHubBlog/public/2020/04/02/leetcode记录-52/index.html","0941ce4acf175dc131cb2680224800d1"],["E:/GitHubBlog/public/2020/04/17/Android第八周作业-每天第一次打开应用更新数据/index.html","03227c9f06605b6cf555266a7783677d"],["E:/GitHubBlog/public/2020/05/21/attention模型/index.html","9115e17064ec0b0cf9df62042ab65db0"],["E:/GitHubBlog/public/2020/05/21/编码器与解码器框架/index.html","1770f57248c7b843f26646513f89b3f0"],["E:/GitHubBlog/public/2020/07/11/思考，快与慢/index.html","f7133699c4b76fe52fdab039b6d9c104"],["E:/GitHubBlog/public/2020/07/13/20200713-0717总结/index.html","4a4e78def39b3603c63a506b65b60221"],["E:/GitHubBlog/public/2020/07/13/ZeroShotCeres Zero-Shot Relation Extraction/index.html","a63cab842fe3c6a5d24478497a0e12e9"],["E:/GitHubBlog/public/2020/07/13/《lingo与excel在数学建模中的应用》/index.html","79e4b7c1b814cb6c2065983b0ffebb61"],["E:/GitHubBlog/public/2020/07/20/Keyphrase Generation A Text Summarization Struggle/index.html","943118c647e9c4c8c83995846b474b91"],["E:/GitHubBlog/public/2020/07/20/TPR/index.html","2f597edc305186a37b3336b1c1815551"],["E:/GitHubBlog/public/2020/07/21/天池NLP赛事-新闻文本分类-Task1-赛题理解/index.html","6cff6fffce061b30a594f7095123bc58"],["E:/GitHubBlog/public/2020/07/22/天池NLP赛事-新闻文本分类-Task2-数据读取与数据分析/index.html","d3a6e485a8895b158244391bcbdb63c5"],["E:/GitHubBlog/public/2020/07/23/Keyphrase Generation for Scientific Document Retrieval/index.html","3f4fbc49abc57e9c8496339697c8e571"],["E:/GitHubBlog/public/2020/07/24/20200718-0724总结/index.html","5b0862ce5a015d6d9e42b8f40856d752"],["E:/GitHubBlog/public/2020/07/25/天池NLP赛事-新闻文本分类-Task3-基于机器学习的文本分类/index.html","d5434b7159108b8cf5b4bd6a0f6bf0ae"],["E:/GitHubBlog/public/2020/07/27/基于深度学习的文本分类/index.html","a61728bc52e034621f8bbfb53d05eb0d"],["E:/GitHubBlog/public/2020/07/31/20200725-0731总结/index.html","e9326f1b1844b3a47256a695ef224bf7"],["E:/GitHubBlog/public/archives/2020/01/index.html","03bcbf01bdc0df52c7310af978ce2003"],["E:/GitHubBlog/public/archives/2020/02/index.html","3ae5343892ec2ff7d5f6114f83c2483b"],["E:/GitHubBlog/public/archives/2020/03/index.html","05f3ee9c0f456067249c224c97b2f83b"],["E:/GitHubBlog/public/archives/2020/04/index.html","a88d0fb4804a549badd175901e036d0d"],["E:/GitHubBlog/public/archives/2020/05/index.html","8fcc2a13f2b2824be5bbdc2557e43c1f"],["E:/GitHubBlog/public/archives/2020/07/index.html","bbdf27f341dee3665454eaaddab33c1c"],["E:/GitHubBlog/public/archives/2020/07/page/2/index.html","03fa722f484deade27b2650782e330f9"],["E:/GitHubBlog/public/archives/2020/index.html","0ed59d328dba17f717e4710c4f306e74"],["E:/GitHubBlog/public/archives/2020/page/2/index.html","0755ee6736e12a77366b57b30f4bdf3b"],["E:/GitHubBlog/public/archives/2020/page/3/index.html","a0e084f0a27943dd14b618d0e0959842"],["E:/GitHubBlog/public/archives/2020/page/4/index.html","d29808e79d1c198e0a5a3919f21a7e34"],["E:/GitHubBlog/public/archives/index.html","c072e62bb1f81050149ae3061ca99639"],["E:/GitHubBlog/public/archives/page/2/index.html","cc818c6a918296427abd58c2334a15fd"],["E:/GitHubBlog/public/archives/page/3/index.html","b88fe9647bdb21ff3ac8aeeb9406c39b"],["E:/GitHubBlog/public/archives/page/4/index.html","5041cf69e4fa41ce8ab6a854611a3ba0"],["E:/GitHubBlog/public/css/404.css","b1bb50e1fabe41adcec483f6427fb3aa"],["E:/GitHubBlog/public/css/index.css","2389364df340948a5b822422956c2c87"],["E:/GitHubBlog/public/css/var.css","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/img/algolia.svg","fd40b88ac5370a5353a50b8175c1f367"],["E:/GitHubBlog/public/img/avatar.png","6cc4a809d23e3d8946a299ae4ce4e4cd"],["E:/GitHubBlog/public/index.html","7b01309fe6dd13cd9852fe251fe0d24f"],["E:/GitHubBlog/public/js/copy.js","45aae54bf2e43ac27ecc41eb5e0bacf7"],["E:/GitHubBlog/public/js/fancybox.js","f84d626654b9bbc05720952b3effe062"],["E:/GitHubBlog/public/js/fireworks.js","35933ce61c158ef9c33b5e089fe757ac"],["E:/GitHubBlog/public/js/head.js","347edd99f8e3921b45fa617e839d8182"],["E:/GitHubBlog/public/js/hexo-theme-melody.js","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/js/katex.js","d971ba8b8dee1120ef66744b89cfd2b1"],["E:/GitHubBlog/public/js/scroll.js","a12ad7e37b9e325f8da3d0523d3757c7"],["E:/GitHubBlog/public/js/search/algolia.js","53160985d32d6fd66d98aa0e05b081ac"],["E:/GitHubBlog/public/js/search/local-search.js","1565b508bd866ed6fbd724a3858af5d8"],["E:/GitHubBlog/public/js/sidebar.js","d24db780974e661198eeb2e45f20a28f"],["E:/GitHubBlog/public/js/third-party/anime.min.js","9b4bbe6deb700e1c3606eab732f5eea5"],["E:/GitHubBlog/public/js/third-party/canvas-ribbon.js","09c181544ddff1db701db02ac30453e0"],["E:/GitHubBlog/public/js/third-party/jquery.fancybox.min.js","3c9fa1c1199cd4f874d855ecb1641335"],["E:/GitHubBlog/public/js/third-party/jquery.min.js","c9f5aeeca3ad37bf2aa006139b935f0a"],["E:/GitHubBlog/public/js/third-party/reveal/head.min.js","aad121203010122e05f1766d92385214"],["E:/GitHubBlog/public/js/third-party/reveal/highlight.min.js","44594243bec43813a16371af8fe7e105"],["E:/GitHubBlog/public/js/third-party/reveal/markdown.min.js","7ec4cef5a7fe3f0bf0eb4dc6d7bca114"],["E:/GitHubBlog/public/js/third-party/reveal/marked.min.js","c2a88705e206d71dc21fdc4445349127"],["E:/GitHubBlog/public/js/third-party/reveal/math.min.js","0a278fee2e57c530ab07f7d2d9ea8d96"],["E:/GitHubBlog/public/js/third-party/reveal/notes.min.js","89a0dfae4d706f9c75b317f686c3aa14"],["E:/GitHubBlog/public/js/third-party/reveal/reveal.min.js","8988419d67efb5fe93e291a357e26ec9"],["E:/GitHubBlog/public/js/third-party/reveal/zoom.min.js","9791f96e63e7d534cba2b67d4bda0419"],["E:/GitHubBlog/public/js/third-party/velocity.min.js","64da069aba987ea0512cf610600a56d1"],["E:/GitHubBlog/public/js/third-party/velocity.ui.min.js","c8ca438424a080620f7b2f4ee4b0fff1"],["E:/GitHubBlog/public/js/transition.js","911db4268f0f6621073afcced9e1bfef"],["E:/GitHubBlog/public/js/utils.js","3ff3423d966a1c351e9867813b3f6d36"],["E:/GitHubBlog/public/page/2/index.html","4d3a13d0b6a14c56291c414e0a7d41c1"],["E:/GitHubBlog/public/page/3/index.html","0ae76cb0514624612aa619f3ebe9a927"],["E:/GitHubBlog/public/page/4/index.html","6ea829426715651eff92c8e3062a8818"],["E:/GitHubBlog/public/tags/Android/index.html","9b21e9809b2d9c1afc6904913651ac3b"],["E:/GitHubBlog/public/tags/NLP/index.html","03327a41aa39848636e6297a8f9b8678"],["E:/GitHubBlog/public/tags/NLP/page/2/index.html","cf73b156b296a7921637d4344fa3d1eb"],["E:/GitHubBlog/public/tags/R/index.html","1e3a5e6943765e143ccd66f52730302b"],["E:/GitHubBlog/public/tags/index.html","6ae4b963fa568b305629368cf825a167"],["E:/GitHubBlog/public/tags/java/index.html","3cc06f0723da96ae12f0ad7f83534a43"],["E:/GitHubBlog/public/tags/leetcode/index.html","6b8e402d240f95ccf6712106c1c4dede"],["E:/GitHubBlog/public/tags/lingo/index.html","e9fad15f1d63872d15aff4b361ddec03"],["E:/GitHubBlog/public/tags/python/index.html","99c8bcd0fe984fc0589a93539deb63ac"],["E:/GitHubBlog/public/tags/总结/index.html","647b69ad1f9ba28ed7969e1b6b4a702c"],["E:/GitHubBlog/public/tags/数据分析/index.html","9f51511b6185053276e51f7d02c123fe"],["E:/GitHubBlog/public/tags/数据结构/index.html","47d1ebf37f954cbe777622cf13d46269"],["E:/GitHubBlog/public/tags/数模/index.html","9072305c87d76e05a264116ac46f060b"],["E:/GitHubBlog/public/tags/比赛/index.html","bb23b84b9243718cf365061a89a842cc"],["E:/GitHubBlog/public/tags/深度学习/index.html","28a3cbcea1253c0c911846a050e6bd7e"],["E:/GitHubBlog/public/tags/爬虫/index.html","d91c107c086b73a481eb9817905bd319"],["E:/GitHubBlog/public/tags/论文/index.html","153ef6f660a902e505559b353883d826"],["E:/GitHubBlog/public/tags/读书笔记/index.html","3f450d483472f97a73d646e713b3c24b"]];
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







