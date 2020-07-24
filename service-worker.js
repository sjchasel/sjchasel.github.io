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

var precacheConfig = [["E:/GitHubBlog/public/2020/01/28/hello-world/index.html","b406c3867b3d5bd1f8511adf6cf3f408"],["E:/GitHubBlog/public/2020/02/08/python爬虫学习总结/index.html","a89a22efa77c2f7e8e6048d798d7ad96"],["E:/GitHubBlog/public/2020/02/19/数据结构第一周-绪论/index.html","d70f0d234e83d316916897092914bc28"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-1/index.html","6be283191d47fdb63dd4a56ac69cd3ad"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-2-创建数据集/index.html","05a20ac0f20d0d906785a6495a7be399"],["E:/GitHubBlog/public/2020/03/01/《R语言实战》学习总结-3-图形初阶/index.html","52e07b9df2cb9f307b5792d6937a8e20"],["E:/GitHubBlog/public/2020/03/02/数据结构第二周-线性结构1/index.html","261f4ecfcb6642abcef84d057911e7a0"],["E:/GitHubBlog/public/2020/03/03/《R语言实战》学习总结-4-基本数据管理/index.html","b3e64d515d6d82b92f00ee6a8ef61b66"],["E:/GitHubBlog/public/2020/03/14/数据结构第三周-线性结构2/index.html","772de7be342e8965f3280fbc77e2438d"],["E:/GitHubBlog/public/2020/03/14/数据结构第四周-栈与队列/index.html","c14462b5d3117da82e03581296f79e3d"],["E:/GitHubBlog/public/2020/03/17/leetcode记录-1、88、20/index.html","f919ba5dca30bb868a6beaa7c817d646"],["E:/GitHubBlog/public/2020/03/20/温度转换程序/index.html","10b2e189f74f46e3bd7f019bc1cffc24"],["E:/GitHubBlog/public/2020/03/22/数据结构第五周-递归/index.html","8c5c5d4e9e9306970252ea554e2b9c3e"],["E:/GitHubBlog/public/2020/03/22/数据结构第六周-串与数组/index.html","60d8697373aeb7c3e0ce1e4787d0e105"],["E:/GitHubBlog/public/2020/04/02/leetcode记录-52/index.html","e7d2c2bb5f808e1406a054df855deec7"],["E:/GitHubBlog/public/2020/04/17/Android第八周作业-每天第一次打开应用更新数据/index.html","61ce1d3da8862efa2785e764acc50849"],["E:/GitHubBlog/public/2020/05/21/attention模型/index.html","8a5658dbaef4cca2ba5698d10d413be1"],["E:/GitHubBlog/public/2020/05/21/编码器与解码器框架/index.html","84aa7e5026153975fca145f1c3ed610c"],["E:/GitHubBlog/public/2020/07/11/思考，快与慢/index.html","8db6c2bcffc39f12482099d49a8b704c"],["E:/GitHubBlog/public/2020/07/13/20200713-0717总结/index.html","da267f15c085b1095258fb4f34da5a0f"],["E:/GitHubBlog/public/2020/07/13/TPR/index.html","843c10a80791222c6d1b630a6293c03e"],["E:/GitHubBlog/public/2020/07/13/ZeroShotCeres Zero-Shot Relation Extraction/index.html","3c2711076eafa60f42977607b95e19de"],["E:/GitHubBlog/public/2020/07/13/《lingo与excel在数学建模中的应用》/index.html","d6e5a92235367bb14d67d5b025989822"],["E:/GitHubBlog/public/2020/07/20/Keyphrase Generation A Text Summarization Struggle/index.html","1149c56d58f9d742baca468eac9d6075"],["E:/GitHubBlog/public/2020/07/21/天池NLP赛事-新闻文本分类-Task1-赛题理解/index.html","9475c3d24d34f3b9755506e5bbeaf4db"],["E:/GitHubBlog/public/2020/07/22/天池NLP赛事-新闻文本分类-Task2-数据读取与数据分析/index.html","5430f3d9876ca1a3d851f16adc337833"],["E:/GitHubBlog/public/2020/07/23/Keyphrase Generation for Scientific Document Retrieval/index.html","7ea1e199e0461c23184dd528b2099119"],["E:/GitHubBlog/public/2020/07/24/20200718-0724总结/index.html","67a10914d1ba618929a325ef5886c285"],["E:/GitHubBlog/public/archives/2020/01/index.html","b0a65eadfd376bb7163b45c0a20f47f9"],["E:/GitHubBlog/public/archives/2020/02/index.html","ca8a452a61212ca61ad788baa69a7977"],["E:/GitHubBlog/public/archives/2020/03/index.html","c158072f5d0cff1d7d792920331d7d20"],["E:/GitHubBlog/public/archives/2020/04/index.html","dacc24a01667e6cb0d52ec3bf4269cd4"],["E:/GitHubBlog/public/archives/2020/05/index.html","c6503f7040fcf271de538846216203d4"],["E:/GitHubBlog/public/archives/2020/07/index.html","e754b96958fce4a6ea36318cc7e2a3a4"],["E:/GitHubBlog/public/archives/2020/index.html","cc463e790a6ba96a7c9da8b58aa2307b"],["E:/GitHubBlog/public/archives/2020/page/2/index.html","43d39a6edaa42f8cf0e6fbce8ce8c732"],["E:/GitHubBlog/public/archives/2020/page/3/index.html","80cb4736f7bd32233ff9fa84c09e13c1"],["E:/GitHubBlog/public/archives/index.html","833722537789ab4037acb522219e6c02"],["E:/GitHubBlog/public/archives/page/2/index.html","b0a07ec4e65d3bf42e85741bbf7224ec"],["E:/GitHubBlog/public/archives/page/3/index.html","9fdaa595e9c4c228ae8c3c534bfa1ed6"],["E:/GitHubBlog/public/css/404.css","b1bb50e1fabe41adcec483f6427fb3aa"],["E:/GitHubBlog/public/css/index.css","2389364df340948a5b822422956c2c87"],["E:/GitHubBlog/public/css/var.css","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/img/algolia.svg","fd40b88ac5370a5353a50b8175c1f367"],["E:/GitHubBlog/public/img/avatar.png","6cc4a809d23e3d8946a299ae4ce4e4cd"],["E:/GitHubBlog/public/index.html","932ac32d7302f0af6dd2a5e778e2c6dc"],["E:/GitHubBlog/public/js/copy.js","45aae54bf2e43ac27ecc41eb5e0bacf7"],["E:/GitHubBlog/public/js/fancybox.js","f84d626654b9bbc05720952b3effe062"],["E:/GitHubBlog/public/js/fireworks.js","35933ce61c158ef9c33b5e089fe757ac"],["E:/GitHubBlog/public/js/head.js","347edd99f8e3921b45fa617e839d8182"],["E:/GitHubBlog/public/js/hexo-theme-melody.js","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/js/katex.js","d971ba8b8dee1120ef66744b89cfd2b1"],["E:/GitHubBlog/public/js/scroll.js","a12ad7e37b9e325f8da3d0523d3757c7"],["E:/GitHubBlog/public/js/search/algolia.js","53160985d32d6fd66d98aa0e05b081ac"],["E:/GitHubBlog/public/js/search/local-search.js","1565b508bd866ed6fbd724a3858af5d8"],["E:/GitHubBlog/public/js/sidebar.js","d24db780974e661198eeb2e45f20a28f"],["E:/GitHubBlog/public/js/third-party/anime.min.js","9b4bbe6deb700e1c3606eab732f5eea5"],["E:/GitHubBlog/public/js/third-party/canvas-ribbon.js","09c181544ddff1db701db02ac30453e0"],["E:/GitHubBlog/public/js/third-party/jquery.fancybox.min.js","3c9fa1c1199cd4f874d855ecb1641335"],["E:/GitHubBlog/public/js/third-party/jquery.min.js","c9f5aeeca3ad37bf2aa006139b935f0a"],["E:/GitHubBlog/public/js/third-party/reveal/head.min.js","aad121203010122e05f1766d92385214"],["E:/GitHubBlog/public/js/third-party/reveal/highlight.min.js","44594243bec43813a16371af8fe7e105"],["E:/GitHubBlog/public/js/third-party/reveal/markdown.min.js","7ec4cef5a7fe3f0bf0eb4dc6d7bca114"],["E:/GitHubBlog/public/js/third-party/reveal/marked.min.js","c2a88705e206d71dc21fdc4445349127"],["E:/GitHubBlog/public/js/third-party/reveal/math.min.js","0a278fee2e57c530ab07f7d2d9ea8d96"],["E:/GitHubBlog/public/js/third-party/reveal/notes.min.js","89a0dfae4d706f9c75b317f686c3aa14"],["E:/GitHubBlog/public/js/third-party/reveal/reveal.min.js","8988419d67efb5fe93e291a357e26ec9"],["E:/GitHubBlog/public/js/third-party/reveal/zoom.min.js","9791f96e63e7d534cba2b67d4bda0419"],["E:/GitHubBlog/public/js/third-party/velocity.min.js","64da069aba987ea0512cf610600a56d1"],["E:/GitHubBlog/public/js/third-party/velocity.ui.min.js","c8ca438424a080620f7b2f4ee4b0fff1"],["E:/GitHubBlog/public/js/transition.js","911db4268f0f6621073afcced9e1bfef"],["E:/GitHubBlog/public/js/utils.js","3ff3423d966a1c351e9867813b3f6d36"],["E:/GitHubBlog/public/page/2/index.html","7c3f5cd9f8fe1a1addaeab4eea2d316b"],["E:/GitHubBlog/public/page/3/index.html","dc3b9dc3f15e6621acfcdaffcd19ab55"],["E:/GitHubBlog/public/tags/Android/index.html","facc1a1ef5ace87bc80b8f37dd0f93ec"],["E:/GitHubBlog/public/tags/NLP/index.html","1aba1ae9a9c52b870fb95fb8b8977c65"],["E:/GitHubBlog/public/tags/R/index.html","7610cd5066f2ed39faa3eee5acb2f73b"],["E:/GitHubBlog/public/tags/index.html","6bdc6ccc5d41b2772da77024d6185f0f"],["E:/GitHubBlog/public/tags/java/index.html","31e8d198f10917b46f4ff2abe2adf8ed"],["E:/GitHubBlog/public/tags/leetcode/index.html","d51c966c601d74ff3630d662ded00404"],["E:/GitHubBlog/public/tags/lingo/index.html","1fed4b2c9911b6e0afbf3ce2be7ddc48"],["E:/GitHubBlog/public/tags/python/index.html","c1127d3f39b448004b61a942d58c3c55"],["E:/GitHubBlog/public/tags/总结/index.html","13c1a4b7746a50f90b99eb53d21a3cc8"],["E:/GitHubBlog/public/tags/数据分析/index.html","c7d9f7c1c99b841cabc58ecc8d649e18"],["E:/GitHubBlog/public/tags/数据结构/index.html","c53c2593f2630f38c5f69f18e16557ed"],["E:/GitHubBlog/public/tags/数模/index.html","3744beb8d7f428d3817a76974abf7d41"],["E:/GitHubBlog/public/tags/比赛/index.html","4dabac3eb07a028a737c17296598a899"],["E:/GitHubBlog/public/tags/深度学习/index.html","16d731cb0252e2a06554c611c18d5527"],["E:/GitHubBlog/public/tags/爬虫/index.html","0d5f54a830a96a9bb55b4310bd68ed29"],["E:/GitHubBlog/public/tags/论文/index.html","fe02c336dc69f95acc3909baf5342956"],["E:/GitHubBlog/public/tags/读书笔记/index.html","bdc0944dfb1af66e659e21e729039b53"]];
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







