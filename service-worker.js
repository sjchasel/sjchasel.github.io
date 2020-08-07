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

var precacheConfig = [["E:/GitHubBlog/public/2020/01/28/hello-world/index.html","ca12abc146a65149fab5fbb142ea8d8c"],["E:/GitHubBlog/public/2020/02/08/python爬虫学习总结/index.html","1390069fa120d0c60bd04b6c789d6308"],["E:/GitHubBlog/public/2020/02/19/数据结构第一周-绪论/index.html","b628fbaab1dcf207f6aae6430a7bbfd0"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-1/index.html","79873810536c5a2e9c0b33aae2c65e71"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-2-创建数据集/index.html","a857051ecdc1aef4549d2075fe0f0801"],["E:/GitHubBlog/public/2020/03/01/《R语言实战》学习总结-3-图形初阶/index.html","cb02a9cd880281e6c6675024c768cd49"],["E:/GitHubBlog/public/2020/03/02/数据结构第二周-线性结构1/index.html","75f3450ef452fab7c2d8b39ff8ddda25"],["E:/GitHubBlog/public/2020/03/03/《R语言实战》学习总结-4-基本数据管理/index.html","1384383453b8ece427028ba4661921a6"],["E:/GitHubBlog/public/2020/03/14/数据结构第三周-线性结构2/index.html","79d8de86bd416d2c13712e70885ea20e"],["E:/GitHubBlog/public/2020/03/14/数据结构第四周-栈与队列/index.html","978255d2b464d1a19f59d253042b96c1"],["E:/GitHubBlog/public/2020/03/17/leetcode记录-1、88、20/index.html","33f0a1f9dbb4f66238239621995610d4"],["E:/GitHubBlog/public/2020/03/20/温度转换程序/index.html","c25d05bbaab60160fdc88d1bb05cb127"],["E:/GitHubBlog/public/2020/03/22/数据结构第五周-递归/index.html","22940573381892b09c4b02f4bc15eabf"],["E:/GitHubBlog/public/2020/03/22/数据结构第六周-串与数组/index.html","89c818cee820a49f61c6a229870ae755"],["E:/GitHubBlog/public/2020/04/02/leetcode记录-52/index.html","a875cf48c09733a3470177236fb0dc53"],["E:/GitHubBlog/public/2020/04/17/Android第八周作业-每天第一次打开应用更新数据/index.html","f5806a3f3d96bc7031e1d86adb890db0"],["E:/GitHubBlog/public/2020/05/21/attention模型/index.html","eb79632df2dd78916cc38f0c182b1631"],["E:/GitHubBlog/public/2020/05/21/编码器与解码器框架/index.html","16ed38adbfe12e2e031e7b8c1d88e36b"],["E:/GitHubBlog/public/2020/07/11/思考，快与慢/index.html","5b9bfab6e250d7c43fc4fb7f77582ed0"],["E:/GitHubBlog/public/2020/07/13/20200713-0717总结/index.html","923bdd28c8826a359a0e2c46a52b1c59"],["E:/GitHubBlog/public/2020/07/13/ZeroShotCeres Zero-Shot Relation Extraction/index.html","c5fab1fb9caef639bf1701c42cec7f3e"],["E:/GitHubBlog/public/2020/07/13/《lingo与excel在数学建模中的应用》/index.html","1dea23f150e89814bab25078375c9ea3"],["E:/GitHubBlog/public/2020/07/20/Keyphrase Generation A Text Summarization Struggle/index.html","f350439d502bd261c36169d8c06a6cab"],["E:/GitHubBlog/public/2020/07/20/TPR/index.html","19bd09ea3bd58ddff5bc9f9a7c01ac54"],["E:/GitHubBlog/public/2020/07/21/天池NLP赛事-新闻文本分类-Task1-赛题理解/index.html","923536d4f1d6a03a3ba0f346b6b2a48d"],["E:/GitHubBlog/public/2020/07/22/天池NLP赛事-新闻文本分类-Task2-数据读取与数据分析/index.html","f611b08346bdb0683765e66fdbc8a35b"],["E:/GitHubBlog/public/2020/07/23/Keyphrase Generation for Scientific Document Retrieval/index.html","1f2d8b94a4c1c183c41cf467d244325a"],["E:/GitHubBlog/public/2020/07/24/20200718-0724总结/index.html","6ef22a47725000d055eaa8ff75a11a96"],["E:/GitHubBlog/public/2020/07/25/天池NLP赛事-新闻文本分类-Task3-基于机器学习的文本分类/index.html","d5e31b0c9432bbc15a0b5cedc2a003c0"],["E:/GitHubBlog/public/2020/07/27/基于深度学习的文本分类/index.html","7680d49e66886776bac7670973958e3c"],["E:/GitHubBlog/public/2020/07/31/20200725-0731总结/index.html","1e7d7b0f015af9d2784c66f007fb364d"],["E:/GitHubBlog/public/2020/07/31/Task5/index.html","7c7109762314e96c740439b1484ca7f1"],["E:/GitHubBlog/public/2020/08/04/基于深度学习的文本分类3/index.html","880a77d1642a8c5c393b4fb9ee314077"],["E:/GitHubBlog/public/2020/08/07/20200801-0807总结/index.html","7221992c4b8d870d86ee3ef3c76cf874"],["E:/GitHubBlog/public/archives/2020/01/index.html","5911b047a19a862053351ee071dab953"],["E:/GitHubBlog/public/archives/2020/02/index.html","f12744abf68b6a96af074bf3127a53b4"],["E:/GitHubBlog/public/archives/2020/03/index.html","17324beb736a474ed9e92c0a755f3d60"],["E:/GitHubBlog/public/archives/2020/04/index.html","bb9a01d5ec3f902ea2cd18d36f0efdf5"],["E:/GitHubBlog/public/archives/2020/05/index.html","fdd2b5361844dd29e4eedeed9f2add55"],["E:/GitHubBlog/public/archives/2020/07/index.html","dc80d87fd97f52bfbbe00471063cf6aa"],["E:/GitHubBlog/public/archives/2020/07/page/2/index.html","31d6d211f496c79e1dea9241a43c7dd1"],["E:/GitHubBlog/public/archives/2020/08/index.html","d872d987c8635ef1ab4637bc3eb8059a"],["E:/GitHubBlog/public/archives/2020/index.html","c35399fd175a643ae251e15d253803ee"],["E:/GitHubBlog/public/archives/2020/page/2/index.html","8dde4dcf018ba635ec2fd13b3ca2eceb"],["E:/GitHubBlog/public/archives/2020/page/3/index.html","72d326a2bc51c8e645000ebaa5ce35cd"],["E:/GitHubBlog/public/archives/2020/page/4/index.html","9cdaa82c5c8fe0a6b110a48f2d63b6d5"],["E:/GitHubBlog/public/archives/index.html","cde910617a550f7da9f518e8a9399f95"],["E:/GitHubBlog/public/archives/page/2/index.html","674a7e0b9aac34f8872ed94b880a6c20"],["E:/GitHubBlog/public/archives/page/3/index.html","e2cf1b652d2acbbb20cf4f60d46a623a"],["E:/GitHubBlog/public/archives/page/4/index.html","e06da6c720f0b6c0c2bf884d94d55274"],["E:/GitHubBlog/public/css/404.css","b1bb50e1fabe41adcec483f6427fb3aa"],["E:/GitHubBlog/public/css/index.css","2389364df340948a5b822422956c2c87"],["E:/GitHubBlog/public/css/var.css","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/img/algolia.svg","fd40b88ac5370a5353a50b8175c1f367"],["E:/GitHubBlog/public/img/avatar.png","6cc4a809d23e3d8946a299ae4ce4e4cd"],["E:/GitHubBlog/public/index.html","16b7eddc11c22dff9bf2c4b34af6f218"],["E:/GitHubBlog/public/js/copy.js","45aae54bf2e43ac27ecc41eb5e0bacf7"],["E:/GitHubBlog/public/js/fancybox.js","f84d626654b9bbc05720952b3effe062"],["E:/GitHubBlog/public/js/fireworks.js","35933ce61c158ef9c33b5e089fe757ac"],["E:/GitHubBlog/public/js/head.js","347edd99f8e3921b45fa617e839d8182"],["E:/GitHubBlog/public/js/hexo-theme-melody.js","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/js/katex.js","d971ba8b8dee1120ef66744b89cfd2b1"],["E:/GitHubBlog/public/js/scroll.js","a12ad7e37b9e325f8da3d0523d3757c7"],["E:/GitHubBlog/public/js/search/algolia.js","53160985d32d6fd66d98aa0e05b081ac"],["E:/GitHubBlog/public/js/search/local-search.js","1565b508bd866ed6fbd724a3858af5d8"],["E:/GitHubBlog/public/js/sidebar.js","d24db780974e661198eeb2e45f20a28f"],["E:/GitHubBlog/public/js/third-party/anime.min.js","9b4bbe6deb700e1c3606eab732f5eea5"],["E:/GitHubBlog/public/js/third-party/canvas-ribbon.js","09c181544ddff1db701db02ac30453e0"],["E:/GitHubBlog/public/js/third-party/jquery.fancybox.min.js","3c9fa1c1199cd4f874d855ecb1641335"],["E:/GitHubBlog/public/js/third-party/jquery.min.js","c9f5aeeca3ad37bf2aa006139b935f0a"],["E:/GitHubBlog/public/js/third-party/reveal/head.min.js","aad121203010122e05f1766d92385214"],["E:/GitHubBlog/public/js/third-party/reveal/highlight.min.js","44594243bec43813a16371af8fe7e105"],["E:/GitHubBlog/public/js/third-party/reveal/markdown.min.js","7ec4cef5a7fe3f0bf0eb4dc6d7bca114"],["E:/GitHubBlog/public/js/third-party/reveal/marked.min.js","c2a88705e206d71dc21fdc4445349127"],["E:/GitHubBlog/public/js/third-party/reveal/math.min.js","0a278fee2e57c530ab07f7d2d9ea8d96"],["E:/GitHubBlog/public/js/third-party/reveal/notes.min.js","89a0dfae4d706f9c75b317f686c3aa14"],["E:/GitHubBlog/public/js/third-party/reveal/reveal.min.js","8988419d67efb5fe93e291a357e26ec9"],["E:/GitHubBlog/public/js/third-party/reveal/zoom.min.js","9791f96e63e7d534cba2b67d4bda0419"],["E:/GitHubBlog/public/js/third-party/velocity.min.js","64da069aba987ea0512cf610600a56d1"],["E:/GitHubBlog/public/js/third-party/velocity.ui.min.js","c8ca438424a080620f7b2f4ee4b0fff1"],["E:/GitHubBlog/public/js/transition.js","911db4268f0f6621073afcced9e1bfef"],["E:/GitHubBlog/public/js/utils.js","3ff3423d966a1c351e9867813b3f6d36"],["E:/GitHubBlog/public/page/2/index.html","0494c28a99bfba4b1815379bc22a63bc"],["E:/GitHubBlog/public/page/3/index.html","34bda938f6123e74d9cf9988a8b2351c"],["E:/GitHubBlog/public/page/4/index.html","1353522566d3c29151acd19b3a6c5e3a"],["E:/GitHubBlog/public/tags/Android/index.html","23191f7176ed23600727981fa8efb0d7"],["E:/GitHubBlog/public/tags/NLP/index.html","bc50c719c80afa8e3ebf6d8e3ccb7b18"],["E:/GitHubBlog/public/tags/NLP/page/2/index.html","d4a8e7d79bba92a23e7b267aeb1c4c92"],["E:/GitHubBlog/public/tags/R/index.html","1f8d9d7d476c5d38b3d4d62bc075ab6d"],["E:/GitHubBlog/public/tags/index.html","fb9b7ded84f9729aef270a33f62f5e1f"],["E:/GitHubBlog/public/tags/java/index.html","d97c0adc866f8005c7c521092ef39603"],["E:/GitHubBlog/public/tags/leetcode/index.html","b756bcd266e09b363d3cf7a632ba8f0d"],["E:/GitHubBlog/public/tags/lingo/index.html","76ca0a967fa61bcac41df995fdf63a29"],["E:/GitHubBlog/public/tags/python/index.html","ec028199c228f945aea4ae81743dd76c"],["E:/GitHubBlog/public/tags/总结/index.html","41194db53b473460e4fc50ee24aab1ae"],["E:/GitHubBlog/public/tags/数据分析/index.html","ba0f4878fdde6dc7bad642c522361e84"],["E:/GitHubBlog/public/tags/数据结构/index.html","e0dc94c30495657ce4818711eec3cc6f"],["E:/GitHubBlog/public/tags/数模/index.html","4c7022a402d256a091dd3d2792eb43bd"],["E:/GitHubBlog/public/tags/比赛/index.html","5fcb24d0a4ff146359f2daa0d9e3124c"],["E:/GitHubBlog/public/tags/深度学习/index.html","796757a11ea61adfc6dfeb7744e5f9bd"],["E:/GitHubBlog/public/tags/爬虫/index.html","7a66c52ef92865dd0aaf00640e9f1bbb"],["E:/GitHubBlog/public/tags/论文/index.html","6f970d3932749b6cf51dd1dda5521647"],["E:/GitHubBlog/public/tags/读书笔记/index.html","b8f82b35c475ec6515c3659a6dd1c947"]];
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







