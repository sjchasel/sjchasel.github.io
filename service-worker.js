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

var precacheConfig = [["E:/GitHubBlog/public/2020/01/28/hello-world/index.html","36c61039e233839e0909920cccdf0573"],["E:/GitHubBlog/public/2020/02/08/python爬虫学习总结/index.html","e55e7b128830488059091dc779ff338a"],["E:/GitHubBlog/public/2020/02/19/数据结构第一周-绪论/index.html","095892bba237fbb3457ca73f217cbbc9"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-1/index.html","18cbd6d296f9e0278e9742951e2db0a9"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-2-创建数据集/index.html","429827f7e27d8aff86d856560628c2fa"],["E:/GitHubBlog/public/2020/03/01/《R语言实战》学习总结-3-图形初阶/index.html","099db05cb1f0e56a3bba169cbe966b62"],["E:/GitHubBlog/public/2020/03/02/数据结构第二周-线性结构1/index.html","c49f11b6b111da5f4165a018d4ba5d4d"],["E:/GitHubBlog/public/2020/03/03/《R语言实战》学习总结-4-基本数据管理/index.html","53bae18f2008c5ca9f9e361e24ce2d6d"],["E:/GitHubBlog/public/2020/03/14/数据结构第三周-线性结构2/index.html","3b7445f14076d67a2dde8a9d431e38f7"],["E:/GitHubBlog/public/2020/03/14/数据结构第四周-栈与队列/index.html","463cacbf54cc1d96c048bf8fd76b4bb0"],["E:/GitHubBlog/public/2020/03/17/leetcode记录-1、88、20/index.html","a7b46b181a96f335f1ae18980b62a0ed"],["E:/GitHubBlog/public/2020/03/20/温度转换程序/index.html","d15b9617663627e66fc5730426e7a72a"],["E:/GitHubBlog/public/2020/03/22/数据结构第五周-递归/index.html","8ea6a6554448c9c79f16cb82145db35d"],["E:/GitHubBlog/public/2020/03/22/数据结构第六周-串与数组/index.html","6297688ecff74bb06750ff2fc6eb6acb"],["E:/GitHubBlog/public/2020/04/02/leetcode记录-52/index.html","93be3203b9bd5f65db8a269c55aa5239"],["E:/GitHubBlog/public/2020/04/17/Android第八周作业-每天第一次打开应用更新数据/index.html","ff0acb150b24aa8841d634e283a6fdba"],["E:/GitHubBlog/public/2020/05/21/attention模型/index.html","888a79ba35bccf553efd78d21b0982e5"],["E:/GitHubBlog/public/2020/05/21/编码器与解码器框架/index.html","aeb0cfd8c775bf216545155c86dfefdb"],["E:/GitHubBlog/public/2020/07/11/思考，快与慢/index.html","d6f4c31dd1b599034d5623d659dc007a"],["E:/GitHubBlog/public/2020/07/13/20200713-0717总结/index.html","f31635e88c31ed9f7a489c56a22dac17"],["E:/GitHubBlog/public/2020/07/13/ZeroShotCeres Zero-Shot Relation Extraction/index.html","54d6e68953e940e75688052f531ba233"],["E:/GitHubBlog/public/2020/07/20/Keyphrase Generation A Text Summarization Struggle/index.html","ff6d7ef72b71b6ddfa5974a49aa9075e"],["E:/GitHubBlog/public/2020/07/20/TPR/index.html","7fe7d5563d57cb59cf926e8dcbf96eee"],["E:/GitHubBlog/public/2020/07/23/Keyphrase Generation for Scientific Document Retrieval/index.html","e5d9f6c5d5ed37919ecce7d0c89c971e"],["E:/GitHubBlog/public/2020/07/24/20200718-0724总结/index.html","f8a5c6fc054f4ff9a275ff2738deb2a3"],["E:/GitHubBlog/public/2020/07/31/20200725-0731总结/index.html","78981645d1e99bf9b40d06608eb9c17f"],["E:/GitHubBlog/public/2020/08/07/20200801-0807总结/index.html","cd38ae2cff7b598cdb29aeae2dc61330"],["E:/GitHubBlog/public/archives/2020/01/index.html","009a4078e6e60f5963e64715fac83909"],["E:/GitHubBlog/public/archives/2020/02/index.html","bbb884215cf248d361bb9e2d2feae5ff"],["E:/GitHubBlog/public/archives/2020/03/index.html","86e1dc34f942623c84220862a6f6ebdb"],["E:/GitHubBlog/public/archives/2020/04/index.html","ef54757c36b86e7c0a8a09588d171f1b"],["E:/GitHubBlog/public/archives/2020/05/index.html","43caa9967e2c3846b340b91603a42f47"],["E:/GitHubBlog/public/archives/2020/07/index.html","f61797b52c6b747d3f00320a029f5b49"],["E:/GitHubBlog/public/archives/2020/08/index.html","d368b3ba10e5c6005d9311761e7ed040"],["E:/GitHubBlog/public/archives/2020/index.html","7156cf22da8374510eb059bec9488a2c"],["E:/GitHubBlog/public/archives/2020/page/2/index.html","f1e634013fd7d9f15ce1aa0721194d53"],["E:/GitHubBlog/public/archives/2020/page/3/index.html","e74fe57c657820ba187a03519afd5536"],["E:/GitHubBlog/public/archives/index.html","9c66ab602ce87026dbe70f8c8c9c0d2c"],["E:/GitHubBlog/public/archives/page/2/index.html","aefc3d449fa89f5b7b6997470566a1a3"],["E:/GitHubBlog/public/archives/page/3/index.html","8080c6d0ef363d3fa58f5a630f29ac5e"],["E:/GitHubBlog/public/css/404.css","b1bb50e1fabe41adcec483f6427fb3aa"],["E:/GitHubBlog/public/css/index.css","2389364df340948a5b822422956c2c87"],["E:/GitHubBlog/public/css/var.css","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/img/algolia.svg","fd40b88ac5370a5353a50b8175c1f367"],["E:/GitHubBlog/public/img/avatar.png","6cc4a809d23e3d8946a299ae4ce4e4cd"],["E:/GitHubBlog/public/index.html","457bec7fb22503f14a29e0396fd246d1"],["E:/GitHubBlog/public/js/copy.js","45aae54bf2e43ac27ecc41eb5e0bacf7"],["E:/GitHubBlog/public/js/fancybox.js","f84d626654b9bbc05720952b3effe062"],["E:/GitHubBlog/public/js/fireworks.js","35933ce61c158ef9c33b5e089fe757ac"],["E:/GitHubBlog/public/js/head.js","347edd99f8e3921b45fa617e839d8182"],["E:/GitHubBlog/public/js/hexo-theme-melody.js","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/js/katex.js","d971ba8b8dee1120ef66744b89cfd2b1"],["E:/GitHubBlog/public/js/scroll.js","a12ad7e37b9e325f8da3d0523d3757c7"],["E:/GitHubBlog/public/js/search/algolia.js","53160985d32d6fd66d98aa0e05b081ac"],["E:/GitHubBlog/public/js/search/local-search.js","1565b508bd866ed6fbd724a3858af5d8"],["E:/GitHubBlog/public/js/sidebar.js","d24db780974e661198eeb2e45f20a28f"],["E:/GitHubBlog/public/js/third-party/anime.min.js","9b4bbe6deb700e1c3606eab732f5eea5"],["E:/GitHubBlog/public/js/third-party/canvas-ribbon.js","09c181544ddff1db701db02ac30453e0"],["E:/GitHubBlog/public/js/third-party/jquery.fancybox.min.js","3c9fa1c1199cd4f874d855ecb1641335"],["E:/GitHubBlog/public/js/third-party/jquery.min.js","c9f5aeeca3ad37bf2aa006139b935f0a"],["E:/GitHubBlog/public/js/third-party/reveal/head.min.js","aad121203010122e05f1766d92385214"],["E:/GitHubBlog/public/js/third-party/reveal/highlight.min.js","44594243bec43813a16371af8fe7e105"],["E:/GitHubBlog/public/js/third-party/reveal/markdown.min.js","7ec4cef5a7fe3f0bf0eb4dc6d7bca114"],["E:/GitHubBlog/public/js/third-party/reveal/marked.min.js","c2a88705e206d71dc21fdc4445349127"],["E:/GitHubBlog/public/js/third-party/reveal/math.min.js","0a278fee2e57c530ab07f7d2d9ea8d96"],["E:/GitHubBlog/public/js/third-party/reveal/notes.min.js","89a0dfae4d706f9c75b317f686c3aa14"],["E:/GitHubBlog/public/js/third-party/reveal/reveal.min.js","8988419d67efb5fe93e291a357e26ec9"],["E:/GitHubBlog/public/js/third-party/reveal/zoom.min.js","9791f96e63e7d534cba2b67d4bda0419"],["E:/GitHubBlog/public/js/third-party/velocity.min.js","64da069aba987ea0512cf610600a56d1"],["E:/GitHubBlog/public/js/third-party/velocity.ui.min.js","c8ca438424a080620f7b2f4ee4b0fff1"],["E:/GitHubBlog/public/js/transition.js","911db4268f0f6621073afcced9e1bfef"],["E:/GitHubBlog/public/js/utils.js","3ff3423d966a1c351e9867813b3f6d36"],["E:/GitHubBlog/public/page/2/index.html","f3285e9608d97b3276b30ed944c3be74"],["E:/GitHubBlog/public/page/3/index.html","c7d8587bc10d838775f8eaad2e8a08d9"],["E:/GitHubBlog/public/tags/Android/index.html","56d3b28e7dfa189c22b2030937cf2546"],["E:/GitHubBlog/public/tags/NLP/index.html","d60f1a04259efae1fdf0347a96fd661a"],["E:/GitHubBlog/public/tags/R/index.html","8a14493d65348d1860150570a65e2e20"],["E:/GitHubBlog/public/tags/index.html","5c3161ba93d2a8bc11184b224e9833e4"],["E:/GitHubBlog/public/tags/java/index.html","f77f5dcd7d03602a3f96cc9ba5ff7e97"],["E:/GitHubBlog/public/tags/leetcode/index.html","6d1a82bea8e751df17c236544827ea18"],["E:/GitHubBlog/public/tags/python/index.html","49d3e7037a6432e8847bcf752f819f46"],["E:/GitHubBlog/public/tags/总结/index.html","8037d1b565072bb39495debf500badce"],["E:/GitHubBlog/public/tags/数据分析/index.html","c91fc791878f87100e89e9404dee218a"],["E:/GitHubBlog/public/tags/数据结构/index.html","d50a25d62355c985546dd1860f2d74e6"],["E:/GitHubBlog/public/tags/深度学习/index.html","2fb05a13ed783c8e7af07a8426ceabec"],["E:/GitHubBlog/public/tags/爬虫/index.html","db9627c0beb7375dd78b63cfe8802bcb"],["E:/GitHubBlog/public/tags/论文/index.html","0634b4f12274657835a01c5173a7872c"],["E:/GitHubBlog/public/tags/读书笔记/index.html","35ae21e8365d78b174d6c29d0e0a0d58"]];
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







