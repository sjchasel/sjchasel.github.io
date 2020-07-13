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

var precacheConfig = [["E:/GitHubBlog/public/2020/01/28/hello-world/index.html","d786f6337c79e1cbff45f0363354ea5d"],["E:/GitHubBlog/public/2020/02/08/python爬虫学习总结/index.html","e7afb4b583bbde090afa09347c35bd2b"],["E:/GitHubBlog/public/2020/02/19/数据结构第一周-绪论/index.html","1d64c7339d63ff4338a02260ae311746"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-1/index.html","3017dfba3b25933c4bad79ee8735526a"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-2-创建数据集/index.html","2e17a12a889830cb08bad7a74031bd11"],["E:/GitHubBlog/public/2020/03/01/《R语言实战》学习总结-3-图形初阶/index.html","058362e9ad2394b8504f011f3f78914c"],["E:/GitHubBlog/public/2020/03/02/数据结构第二周-线性结构1/index.html","e07719fb727b51f0651e77983857197f"],["E:/GitHubBlog/public/2020/03/03/《R语言实战》学习总结-4-基本数据管理/index.html","432e264f7b642bf74f2f6fa14b179d42"],["E:/GitHubBlog/public/2020/03/14/数据结构第三周-线性结构2/index.html","153a0137efde70972ac54b486779d8f3"],["E:/GitHubBlog/public/2020/03/14/数据结构第四周-栈与队列/index.html","755617d529a357f00554407f2fb8f6b6"],["E:/GitHubBlog/public/2020/03/17/leetcode记录-1、88、20/index.html","2b220ad29838916ad5c66a8e8cd1206d"],["E:/GitHubBlog/public/2020/03/20/python爬虫第一周/index.html","2bb8ca6e8b43e0e4f89152749540bd0d"],["E:/GitHubBlog/public/2020/03/20/温度转换程序/index.html","730eaec561df985d4f7a13ecca0dec8d"],["E:/GitHubBlog/public/2020/03/22/数据结构第五周-递归/index.html","5e825362c164bf8365694d0bcc476684"],["E:/GitHubBlog/public/2020/03/22/数据结构第六周-串与数组/index.html","aa7c2d91466309d467f2fc3baad384ce"],["E:/GitHubBlog/public/2020/04/02/leetcode记录-52/index.html","df56b7ce7afde7a6b3e915a56c6dafb1"],["E:/GitHubBlog/public/2020/04/17/Android第八周作业-每天第一次打开应用更新数据/index.html","c7137e4be5760034e1884f1e7c8edb04"],["E:/GitHubBlog/public/2020/04/24/8论文/index.html","3c781ba656b856934431887bd08cb3c0"],["E:/GitHubBlog/public/2020/04/25/10论文/index.html","c8cc9ffa18a35cd6ab6318dcf6f25173"],["E:/GitHubBlog/public/2020/04/25/4论文/index.html","a825b68cead0514607d81c8dac3058e4"],["E:/GitHubBlog/public/2020/05/21/attention模型/index.html","4c4b6537c69f37d2108016d758be68ff"],["E:/GitHubBlog/public/2020/05/21/编码器与解码器框架/index.html","8fa6a2c708e0d26137eef0bae820a696"],["E:/GitHubBlog/public/2020/07/11/#信息#利用python进行数据分析/index.html","9c02369553b1a108e8b08c0e6ff94487"],["E:/GitHubBlog/public/2020/07/11/#信息#长尾理论/index.html","b1f3f3d0c8f900774759b413e84446e4"],["E:/GitHubBlog/public/2020/07/11/#经管#思考，快与慢/index.html","0c3737b6cacdc5781b81fd9a27487b86"],["E:/GitHubBlog/public/2020/07/11/精益数据分析/index.html","3b4d8de59d7ecf9cd770711b33b4c09d"],["E:/GitHubBlog/public/archives/2020/01/index.html","816db90505b3d6393a5595c876f4d19d"],["E:/GitHubBlog/public/archives/2020/02/index.html","3e5805df75924e1b5ecc325ae150b775"],["E:/GitHubBlog/public/archives/2020/03/index.html","2e56f0d27e9d3e7c1ce5724be9455039"],["E:/GitHubBlog/public/archives/2020/04/index.html","dfb12bdc38d1130c024e43773c7519b6"],["E:/GitHubBlog/public/archives/2020/05/index.html","b2c5cb46733987ab6cbb8a97a5e17156"],["E:/GitHubBlog/public/archives/2020/07/index.html","90df096b7a37ae860e65cb31402afec9"],["E:/GitHubBlog/public/archives/2020/index.html","2ba878121c30acedde332f05cf328221"],["E:/GitHubBlog/public/archives/2020/page/2/index.html","479ba685498bcb4bbb958fcfded1f3da"],["E:/GitHubBlog/public/archives/2020/page/3/index.html","095ba8519f62d6280330af0b13977b7c"],["E:/GitHubBlog/public/archives/index.html","dcea0eda5ff0db5fe16c4faf00e2b492"],["E:/GitHubBlog/public/archives/page/2/index.html","1f96a1083f16a4709a8625a285a34461"],["E:/GitHubBlog/public/archives/page/3/index.html","3eeef30b0c925a5cf4cc0aafe7936d4f"],["E:/GitHubBlog/public/css/404.css","b1bb50e1fabe41adcec483f6427fb3aa"],["E:/GitHubBlog/public/css/index.css","2389364df340948a5b822422956c2c87"],["E:/GitHubBlog/public/css/var.css","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/img/algolia.svg","fd40b88ac5370a5353a50b8175c1f367"],["E:/GitHubBlog/public/img/avatar.png","6cc4a809d23e3d8946a299ae4ce4e4cd"],["E:/GitHubBlog/public/index.html","34f7e5e7689e9bf3512de5eabf77546a"],["E:/GitHubBlog/public/js/copy.js","45aae54bf2e43ac27ecc41eb5e0bacf7"],["E:/GitHubBlog/public/js/fancybox.js","f84d626654b9bbc05720952b3effe062"],["E:/GitHubBlog/public/js/fireworks.js","35933ce61c158ef9c33b5e089fe757ac"],["E:/GitHubBlog/public/js/head.js","347edd99f8e3921b45fa617e839d8182"],["E:/GitHubBlog/public/js/hexo-theme-melody.js","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/js/katex.js","d971ba8b8dee1120ef66744b89cfd2b1"],["E:/GitHubBlog/public/js/scroll.js","a12ad7e37b9e325f8da3d0523d3757c7"],["E:/GitHubBlog/public/js/search/algolia.js","53160985d32d6fd66d98aa0e05b081ac"],["E:/GitHubBlog/public/js/search/local-search.js","1565b508bd866ed6fbd724a3858af5d8"],["E:/GitHubBlog/public/js/sidebar.js","d24db780974e661198eeb2e45f20a28f"],["E:/GitHubBlog/public/js/third-party/anime.min.js","9b4bbe6deb700e1c3606eab732f5eea5"],["E:/GitHubBlog/public/js/third-party/canvas-ribbon.js","09c181544ddff1db701db02ac30453e0"],["E:/GitHubBlog/public/js/third-party/jquery.fancybox.min.js","3c9fa1c1199cd4f874d855ecb1641335"],["E:/GitHubBlog/public/js/third-party/jquery.min.js","c9f5aeeca3ad37bf2aa006139b935f0a"],["E:/GitHubBlog/public/js/third-party/reveal/head.min.js","aad121203010122e05f1766d92385214"],["E:/GitHubBlog/public/js/third-party/reveal/highlight.min.js","44594243bec43813a16371af8fe7e105"],["E:/GitHubBlog/public/js/third-party/reveal/markdown.min.js","7ec4cef5a7fe3f0bf0eb4dc6d7bca114"],["E:/GitHubBlog/public/js/third-party/reveal/marked.min.js","c2a88705e206d71dc21fdc4445349127"],["E:/GitHubBlog/public/js/third-party/reveal/math.min.js","0a278fee2e57c530ab07f7d2d9ea8d96"],["E:/GitHubBlog/public/js/third-party/reveal/notes.min.js","89a0dfae4d706f9c75b317f686c3aa14"],["E:/GitHubBlog/public/js/third-party/reveal/reveal.min.js","8988419d67efb5fe93e291a357e26ec9"],["E:/GitHubBlog/public/js/third-party/reveal/zoom.min.js","9791f96e63e7d534cba2b67d4bda0419"],["E:/GitHubBlog/public/js/third-party/velocity.min.js","64da069aba987ea0512cf610600a56d1"],["E:/GitHubBlog/public/js/third-party/velocity.ui.min.js","c8ca438424a080620f7b2f4ee4b0fff1"],["E:/GitHubBlog/public/js/transition.js","911db4268f0f6621073afcced9e1bfef"],["E:/GitHubBlog/public/js/utils.js","3ff3423d966a1c351e9867813b3f6d36"],["E:/GitHubBlog/public/page/2/index.html","d6cce8bcb2e855216ec65ad4ab684862"],["E:/GitHubBlog/public/page/3/index.html","fccc64d10897ece98f25d1c5feedc158"],["E:/GitHubBlog/public/tags/Android/index.html","5b730c89de963b1e48de06cd26c14a32"],["E:/GitHubBlog/public/tags/R/index.html","2e0451a77e1ad036d7f111e4b76f7114"],["E:/GitHubBlog/public/tags/index.html","4ded5c4f811c6775f12225cd524bba98"],["E:/GitHubBlog/public/tags/java/index.html","92595e534270865567cf72308a4356b7"],["E:/GitHubBlog/public/tags/leetcode/index.html","2474193cdc2b1db90b3ce51c6c6945fe"],["E:/GitHubBlog/public/tags/python/index.html","6413f015fef807b2c334c90acf1a7650"],["E:/GitHubBlog/public/tags/数据分析/index.html","a2c84e7749180c40dd12a096a3187c02"],["E:/GitHubBlog/public/tags/数据结构/index.html","fba6db48d9a0f4eb7d1e5b9efa4d1326"],["E:/GitHubBlog/public/tags/深度学习/index.html","311795a0c90a5b94795efb5b69b293b8"],["E:/GitHubBlog/public/tags/爬虫/index.html","3fd1bf888343d8738ea43da8ad6731b6"],["E:/GitHubBlog/public/tags/知识图谱/index.html","d1db022d7971489434ee0f6cf58de66e"],["E:/GitHubBlog/public/tags/论文/index.html","db51405a951a3f0bd1c36d3ca45e00c5"],["E:/GitHubBlog/public/tags/读书笔记/index.html","e41b44825a259fd0a29d50c15cd98899"]];
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







