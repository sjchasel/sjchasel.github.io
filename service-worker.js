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

var precacheConfig = [["E:/GitHubBlog/public/2020/01/28/hello-world/index.html","62f4553ed4dff7b84fbfc4797f868deb"],["E:/GitHubBlog/public/2020/02/08/python爬虫学习总结/index.html","b039884f65622524b85c77d5ceecad10"],["E:/GitHubBlog/public/2020/02/19/数据结构第一周-绪论/index.html","64bc932c0865a32905a8cd6ad73c0024"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-1/index.html","0c9f47f32db1fe95f364a775e3afe6f1"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-2-创建数据集/index.html","91b103ff54f62fb4326ad0ed238412ae"],["E:/GitHubBlog/public/2020/03/01/《R语言实战》学习总结-3-图形初阶/index.html","62b56954b9e9188efc702ed3859c18db"],["E:/GitHubBlog/public/2020/03/02/数据结构第二周-线性结构1/index.html","0c852b2c91d59b3bdadd5d75ce59f070"],["E:/GitHubBlog/public/2020/03/03/《R语言实战》学习总结-4-基本数据管理/index.html","3eb6085a7c501fcdfdfbe6868c7282a2"],["E:/GitHubBlog/public/2020/03/14/数据结构第三周-线性结构2/index.html","4530b3cb7d43d0bec001361af27f74c0"],["E:/GitHubBlog/public/2020/03/14/数据结构第四周-栈与队列/index.html","e905bb2531df6a7d9b255be4efb8a2e8"],["E:/GitHubBlog/public/2020/03/17/leetcode记录-1、88、20/index.html","875520987b00c41640e707f9e16ff6a9"],["E:/GitHubBlog/public/2020/03/20/温度转换程序/index.html","69376fc16c466df10442179f5210628a"],["E:/GitHubBlog/public/2020/03/22/数据结构第五周-递归/index.html","ab75f404c35bf3b308ec74fa944e59a4"],["E:/GitHubBlog/public/2020/03/22/数据结构第六周-串与数组/index.html","2099b754029b2eb3d8fd27a3eac2502b"],["E:/GitHubBlog/public/2020/04/02/leetcode记录-52/index.html","987a3d11cd78b0664f0100f29f6a8a6c"],["E:/GitHubBlog/public/2020/04/17/Android第八周作业-每天第一次打开应用更新数据/index.html","dabe7a8dd02313e3b6a8c78fb45f461d"],["E:/GitHubBlog/public/2020/05/21/attention模型/index.html","0e914588036d96288239456aa2d9b1f5"],["E:/GitHubBlog/public/2020/05/21/编码器与解码器框架/index.html","d32d94996ab4426efe2cabc6f8b057b6"],["E:/GitHubBlog/public/2020/07/11/思考，快与慢/index.html","45b8af54b88b4ae7c5d3da2ee2a74cc2"],["E:/GitHubBlog/public/2020/07/13/20200713-0717总结/index.html","15a4367892981cdd7b99a468e034068a"],["E:/GitHubBlog/public/2020/07/13/TPR/index.html","46e5209575ea33076139ce26a6a3f96b"],["E:/GitHubBlog/public/2020/07/13/ZeroShotCeres Zero-Shot Relation Extraction/index.html","fa82a1d2fa2d7bf24ed5615d93c8d660"],["E:/GitHubBlog/public/2020/07/13/《lingo与excel在数学建模中的应用》/index.html","f6f1b15c3daf87c42bc2ec41bcd64e38"],["E:/GitHubBlog/public/2020/07/20/Keyphrase Generation A Text Summarization Struggle/index.html","64a05e3902db9c43ddd434d65c71956a"],["E:/GitHubBlog/public/2020/07/21/天池NLP赛事-新闻文本分类-Task1-赛题理解/index.html","092d5158e9d10e43d46946f162c76d73"],["E:/GitHubBlog/public/2020/07/22/天池NLP赛事-新闻文本分类-Task2-数据读取与数据分析/index.html","ec8dcca765807052facb5681c88c331b"],["E:/GitHubBlog/public/archives/2020/01/index.html","6a64c0e0585e1e95f64d818c2ab25569"],["E:/GitHubBlog/public/archives/2020/02/index.html","7ca8fdd0e196b4947b87c1be1fe9b2c1"],["E:/GitHubBlog/public/archives/2020/03/index.html","8fa64d94069c79910ee66a3e95b208d3"],["E:/GitHubBlog/public/archives/2020/04/index.html","befbc548c550e49515b81d0cb3720ab9"],["E:/GitHubBlog/public/archives/2020/05/index.html","c4254263ed511d3ca5637ea59322325c"],["E:/GitHubBlog/public/archives/2020/07/index.html","b528c3393eddce7e1389505c00715b71"],["E:/GitHubBlog/public/archives/2020/index.html","30ec615c2e63003d1f3092ef5b6344b7"],["E:/GitHubBlog/public/archives/2020/page/2/index.html","10959f9794f2a1f9c2cdcf22eff88284"],["E:/GitHubBlog/public/archives/2020/page/3/index.html","029d381bd253ef5459e88e0c1d423d9c"],["E:/GitHubBlog/public/archives/index.html","170bc89c81617ced9721404bcbfb359e"],["E:/GitHubBlog/public/archives/page/2/index.html","2ea8f51ad406305078b867a7b199945e"],["E:/GitHubBlog/public/archives/page/3/index.html","2f59167d7d67d76106f24bb28224a4f1"],["E:/GitHubBlog/public/css/404.css","b1bb50e1fabe41adcec483f6427fb3aa"],["E:/GitHubBlog/public/css/index.css","2389364df340948a5b822422956c2c87"],["E:/GitHubBlog/public/css/var.css","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/img/algolia.svg","fd40b88ac5370a5353a50b8175c1f367"],["E:/GitHubBlog/public/img/avatar.png","6cc4a809d23e3d8946a299ae4ce4e4cd"],["E:/GitHubBlog/public/index.html","a7d98d219763e13f70cb0317fb387530"],["E:/GitHubBlog/public/js/copy.js","45aae54bf2e43ac27ecc41eb5e0bacf7"],["E:/GitHubBlog/public/js/fancybox.js","f84d626654b9bbc05720952b3effe062"],["E:/GitHubBlog/public/js/fireworks.js","35933ce61c158ef9c33b5e089fe757ac"],["E:/GitHubBlog/public/js/head.js","347edd99f8e3921b45fa617e839d8182"],["E:/GitHubBlog/public/js/hexo-theme-melody.js","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/js/katex.js","d971ba8b8dee1120ef66744b89cfd2b1"],["E:/GitHubBlog/public/js/scroll.js","a12ad7e37b9e325f8da3d0523d3757c7"],["E:/GitHubBlog/public/js/search/algolia.js","53160985d32d6fd66d98aa0e05b081ac"],["E:/GitHubBlog/public/js/search/local-search.js","1565b508bd866ed6fbd724a3858af5d8"],["E:/GitHubBlog/public/js/sidebar.js","d24db780974e661198eeb2e45f20a28f"],["E:/GitHubBlog/public/js/third-party/anime.min.js","9b4bbe6deb700e1c3606eab732f5eea5"],["E:/GitHubBlog/public/js/third-party/canvas-ribbon.js","09c181544ddff1db701db02ac30453e0"],["E:/GitHubBlog/public/js/third-party/jquery.fancybox.min.js","3c9fa1c1199cd4f874d855ecb1641335"],["E:/GitHubBlog/public/js/third-party/jquery.min.js","c9f5aeeca3ad37bf2aa006139b935f0a"],["E:/GitHubBlog/public/js/third-party/reveal/head.min.js","aad121203010122e05f1766d92385214"],["E:/GitHubBlog/public/js/third-party/reveal/highlight.min.js","44594243bec43813a16371af8fe7e105"],["E:/GitHubBlog/public/js/third-party/reveal/markdown.min.js","7ec4cef5a7fe3f0bf0eb4dc6d7bca114"],["E:/GitHubBlog/public/js/third-party/reveal/marked.min.js","c2a88705e206d71dc21fdc4445349127"],["E:/GitHubBlog/public/js/third-party/reveal/math.min.js","0a278fee2e57c530ab07f7d2d9ea8d96"],["E:/GitHubBlog/public/js/third-party/reveal/notes.min.js","89a0dfae4d706f9c75b317f686c3aa14"],["E:/GitHubBlog/public/js/third-party/reveal/reveal.min.js","8988419d67efb5fe93e291a357e26ec9"],["E:/GitHubBlog/public/js/third-party/reveal/zoom.min.js","9791f96e63e7d534cba2b67d4bda0419"],["E:/GitHubBlog/public/js/third-party/velocity.min.js","64da069aba987ea0512cf610600a56d1"],["E:/GitHubBlog/public/js/third-party/velocity.ui.min.js","c8ca438424a080620f7b2f4ee4b0fff1"],["E:/GitHubBlog/public/js/transition.js","911db4268f0f6621073afcced9e1bfef"],["E:/GitHubBlog/public/js/utils.js","3ff3423d966a1c351e9867813b3f6d36"],["E:/GitHubBlog/public/page/2/index.html","5ad1fd29a42cee2819325c6039905a10"],["E:/GitHubBlog/public/page/3/index.html","c4f596926208e8575fb4d4982c9d1614"],["E:/GitHubBlog/public/tags/Android/index.html","c4b16fa9e163103b99d24640ce825135"],["E:/GitHubBlog/public/tags/NLP/index.html","d51879138a118641b9bc97c5f4364435"],["E:/GitHubBlog/public/tags/R/index.html","674eb09063e2f0ac07c5bd22e2934bd2"],["E:/GitHubBlog/public/tags/index.html","15c9f6adbd83267a460bfaa631469abb"],["E:/GitHubBlog/public/tags/java/index.html","7a389bc77779acd8caa3aae61342cb80"],["E:/GitHubBlog/public/tags/leetcode/index.html","5b30980b177544fb22d804747377ca2f"],["E:/GitHubBlog/public/tags/lingo/index.html","68b26e6da466391c225d9279d38e825f"],["E:/GitHubBlog/public/tags/python/index.html","d2be493dea3b99db1bb2e6ea30ca55b7"],["E:/GitHubBlog/public/tags/总结/index.html","1d991a7da8ca3225b0a2292412108818"],["E:/GitHubBlog/public/tags/数据分析/index.html","62d8803edaf4a532cb04e38e72088804"],["E:/GitHubBlog/public/tags/数据结构/index.html","697e71c70d10e1fa420a8ab05d56c87d"],["E:/GitHubBlog/public/tags/数模/index.html","1e148b755d246948c7543818f25805b3"],["E:/GitHubBlog/public/tags/比赛/index.html","4c9efbb2ee7f426a0fe2a57ee8c9c43b"],["E:/GitHubBlog/public/tags/深度学习/index.html","b256115a77c7d07533b26cb5384e1a2f"],["E:/GitHubBlog/public/tags/爬虫/index.html","cf2aba4c2ad5d40064bce0bb9874b4dc"],["E:/GitHubBlog/public/tags/论文/index.html","c2d056280ecfca453ad199fef6d4add6"],["E:/GitHubBlog/public/tags/读书笔记/index.html","13c94684d956c6100114ff8850d34f97"]];
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







