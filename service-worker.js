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

var precacheConfig = [["E:/GitHubBlog/public/2020/01/28/hello-world/index.html","ec6311c71e31a4044a7e4767bf71dddf"],["E:/GitHubBlog/public/2020/02/08/python爬虫学习总结/index.html","ad75f4e87a9016b0ef6ab83065e2081e"],["E:/GitHubBlog/public/2020/02/19/数据结构第一周-绪论/index.html","174e612d18fde7490222535610d77944"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-1/index.html","7450a972cb905fcc8a911cd6bd47c58b"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-2-创建数据集/index.html","56e2f97e9c2e5215ed382d845651e31c"],["E:/GitHubBlog/public/2020/03/01/《R语言实战》学习总结-3-图形初阶/index.html","0f69575aa4934e3062e9fe7a198b71e6"],["E:/GitHubBlog/public/2020/03/01/黄老师的打油诗记录/index.html","d5b8e68095b99ecfab77595e5c98ec6f"],["E:/GitHubBlog/public/2020/03/02/数据结构第二周-线性结构1/index.html","935991066d3988aaaa67183da3267b44"],["E:/GitHubBlog/public/2020/03/03/《R语言实战》学习总结-4-基本数据管理/index.html","846920b74f467b9bef62886a0a7b21f9"],["E:/GitHubBlog/public/2020/03/14/数据结构第三周-线性结构2/index.html","0f2e8161d4a00cdabfe12b275818e833"],["E:/GitHubBlog/public/2020/03/14/数据结构第四周-栈与队列/index.html","8fd6bd193ca076d549450b7a8decef88"],["E:/GitHubBlog/public/2020/03/17/leetcode记录-1、88、20/index.html","b86b6ce299c6a720bd1894515b0d6925"],["E:/GitHubBlog/public/2020/03/20/python爬虫第一周/index.html","3c9e627628f3f9f11a39a02df311e594"],["E:/GitHubBlog/public/2020/03/20/温度转换程序/index.html","f9671e070e922d5e2487daab0c7ac85c"],["E:/GitHubBlog/public/2020/03/22/数据结构第五周-递归/index.html","c8afc929ee9ddbf61dd0be67354981b5"],["E:/GitHubBlog/public/2020/03/22/数据结构第六周-串与数组/index.html","39bfe550c7739645536618a24a809aa8"],["E:/GitHubBlog/public/2020/04/02/leetcode记录-52/index.html","a3d9762441fe632b5d7570b4ac6ded3d"],["E:/GitHubBlog/public/2020/04/17/Android第八周作业-每天第一次打开应用更新数据/index.html","df35abe33fe5d414b1e4af67c14d7a69"],["E:/GitHubBlog/public/2020/04/24/8论文/index.html","ab76d30b0468c7d1d09fb69c7d26602d"],["E:/GitHubBlog/public/2020/04/25/10论文/index.html","997a2a298890c02b5d5721b7881a5e70"],["E:/GitHubBlog/public/2020/04/25/4论文/index.html","a1ee9b605ed1ede9ac081fe4acedfc64"],["E:/GitHubBlog/public/2020/05/21/attention模型/index.html","0f481ca508ed902fed81bab89b2925b3"],["E:/GitHubBlog/public/2020/05/21/编码器与解码器框架/index.html","ab7aa1b3ff6f68bab17cfbc642982835"],["E:/GitHubBlog/public/archives/2020/01/index.html","2a2ee13f2691080a1256435e785a09d5"],["E:/GitHubBlog/public/archives/2020/02/index.html","8f200f5433768aacf976c4c3c52b43b9"],["E:/GitHubBlog/public/archives/2020/03/index.html","1f906d23181470ba486d72e1bf3f6d1c"],["E:/GitHubBlog/public/archives/2020/03/page/2/index.html","9f868c90f47810be3ea5da9b5adadacd"],["E:/GitHubBlog/public/archives/2020/04/index.html","44e441b423c3542498de3b32baf0da8a"],["E:/GitHubBlog/public/archives/2020/05/index.html","93237a0a31529b7030f071333b5fd535"],["E:/GitHubBlog/public/archives/2020/index.html","4d7077e80b454339e96a2ca41a21428b"],["E:/GitHubBlog/public/archives/2020/page/2/index.html","d0ac39240f7aa51622a97088caf5e12b"],["E:/GitHubBlog/public/archives/2020/page/3/index.html","1e77d97fde90f78cdfa3a11613fd6e3b"],["E:/GitHubBlog/public/archives/index.html","944f9716098f98d3f3cd7d32d41b8022"],["E:/GitHubBlog/public/archives/page/2/index.html","b44cb49b8a2d63654572e22e13967a17"],["E:/GitHubBlog/public/archives/page/3/index.html","94b2e97a5301bf8cbb3610aef7353685"],["E:/GitHubBlog/public/css/404.css","b1bb50e1fabe41adcec483f6427fb3aa"],["E:/GitHubBlog/public/css/index.css","2389364df340948a5b822422956c2c87"],["E:/GitHubBlog/public/css/var.css","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/img/algolia.svg","fd40b88ac5370a5353a50b8175c1f367"],["E:/GitHubBlog/public/img/avatar.png","6cc4a809d23e3d8946a299ae4ce4e4cd"],["E:/GitHubBlog/public/index.html","fac2dfab063837cfe813b9e29ffc90d8"],["E:/GitHubBlog/public/js/copy.js","45aae54bf2e43ac27ecc41eb5e0bacf7"],["E:/GitHubBlog/public/js/fancybox.js","f84d626654b9bbc05720952b3effe062"],["E:/GitHubBlog/public/js/fireworks.js","35933ce61c158ef9c33b5e089fe757ac"],["E:/GitHubBlog/public/js/head.js","347edd99f8e3921b45fa617e839d8182"],["E:/GitHubBlog/public/js/hexo-theme-melody.js","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/js/katex.js","d971ba8b8dee1120ef66744b89cfd2b1"],["E:/GitHubBlog/public/js/scroll.js","a12ad7e37b9e325f8da3d0523d3757c7"],["E:/GitHubBlog/public/js/search/algolia.js","53160985d32d6fd66d98aa0e05b081ac"],["E:/GitHubBlog/public/js/search/local-search.js","1565b508bd866ed6fbd724a3858af5d8"],["E:/GitHubBlog/public/js/sidebar.js","d24db780974e661198eeb2e45f20a28f"],["E:/GitHubBlog/public/js/third-party/anime.min.js","9b4bbe6deb700e1c3606eab732f5eea5"],["E:/GitHubBlog/public/js/third-party/canvas-ribbon.js","09c181544ddff1db701db02ac30453e0"],["E:/GitHubBlog/public/js/third-party/jquery.fancybox.min.js","3c9fa1c1199cd4f874d855ecb1641335"],["E:/GitHubBlog/public/js/third-party/jquery.min.js","c9f5aeeca3ad37bf2aa006139b935f0a"],["E:/GitHubBlog/public/js/third-party/reveal/head.min.js","aad121203010122e05f1766d92385214"],["E:/GitHubBlog/public/js/third-party/reveal/highlight.min.js","44594243bec43813a16371af8fe7e105"],["E:/GitHubBlog/public/js/third-party/reveal/markdown.min.js","7ec4cef5a7fe3f0bf0eb4dc6d7bca114"],["E:/GitHubBlog/public/js/third-party/reveal/marked.min.js","c2a88705e206d71dc21fdc4445349127"],["E:/GitHubBlog/public/js/third-party/reveal/math.min.js","0a278fee2e57c530ab07f7d2d9ea8d96"],["E:/GitHubBlog/public/js/third-party/reveal/notes.min.js","89a0dfae4d706f9c75b317f686c3aa14"],["E:/GitHubBlog/public/js/third-party/reveal/reveal.min.js","8988419d67efb5fe93e291a357e26ec9"],["E:/GitHubBlog/public/js/third-party/reveal/zoom.min.js","9791f96e63e7d534cba2b67d4bda0419"],["E:/GitHubBlog/public/js/third-party/velocity.min.js","64da069aba987ea0512cf610600a56d1"],["E:/GitHubBlog/public/js/third-party/velocity.ui.min.js","c8ca438424a080620f7b2f4ee4b0fff1"],["E:/GitHubBlog/public/js/transition.js","911db4268f0f6621073afcced9e1bfef"],["E:/GitHubBlog/public/js/utils.js","3ff3423d966a1c351e9867813b3f6d36"],["E:/GitHubBlog/public/page/2/index.html","56f5925d3aef90d1145566f91ceb9a9e"],["E:/GitHubBlog/public/page/3/index.html","4a4ecb2f49c4e23cf8ef327fb436a3ab"],["E:/GitHubBlog/public/tags/Android/index.html","089a388fe06d77129843504ae3a21e83"],["E:/GitHubBlog/public/tags/R/index.html","61b71f7e3100e6d21920643eb1532fbf"],["E:/GitHubBlog/public/tags/index.html","d475d58d7dc7376625edf53cbd718802"],["E:/GitHubBlog/public/tags/java/index.html","55b0d65e4ee3b2a92250e0adf6764225"],["E:/GitHubBlog/public/tags/leetcode/index.html","79fbd8b159f2f7386ff81eb0a710e7c9"],["E:/GitHubBlog/public/tags/python/index.html","2432269ec0a4df0f315f47e869761d98"],["E:/GitHubBlog/public/tags/数据分析/index.html","d39f07e00e3bf1e09cdb74b0899f22ed"],["E:/GitHubBlog/public/tags/数据结构/index.html","146f400bf60155e0522dbe63540cf617"],["E:/GitHubBlog/public/tags/深度学习/index.html","2799125a93d06f4afb4e350af0d55b84"],["E:/GitHubBlog/public/tags/爬虫/index.html","4ef500aa2b69b7d17db056ec79af6e1f"],["E:/GitHubBlog/public/tags/知识图谱/index.html","147493ecb2f660821275aca71423827c"],["E:/GitHubBlog/public/tags/论文/index.html","d7939c1b4fba02f235a77262fc7f1ec1"]];
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







