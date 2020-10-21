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

var precacheConfig = [["E:/GitHubBlog/public/2020/01/28/hello-world/index.html","ee734d323a8e6b0dca00de66bea23816"],["E:/GitHubBlog/public/2020/02/08/python爬虫学习总结/index.html","d93927492e7d225f34d32db4497dc91c"],["E:/GitHubBlog/public/2020/02/19/数据结构第一周-绪论/index.html","a84dfa91d2f0e987657e5003e4bb8eca"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-1/index.html","7df5439b1b5561727cb877f39e8e8410"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-2-创建数据集/index.html","7428835d49e89a6e47f2f8446d67528f"],["E:/GitHubBlog/public/2020/03/01/《R语言实战》学习总结-3-图形初阶/index.html","bd28cfcad1b9141d66749e5d6bdbdf63"],["E:/GitHubBlog/public/2020/03/02/数据结构第二周-线性结构1/index.html","76566ca89ed6dca7ba9df2110bd14a0e"],["E:/GitHubBlog/public/2020/03/03/《R语言实战》学习总结-4-基本数据管理/index.html","7489df99961242b0b37c2a328be4323b"],["E:/GitHubBlog/public/2020/03/14/数据结构第三周-线性结构2/index.html","306fc2c127f05b4ce0baf3289d748395"],["E:/GitHubBlog/public/2020/03/14/数据结构第四周-栈与队列/index.html","5200b5fdff11bb5d2ccb139358a4416d"],["E:/GitHubBlog/public/2020/03/17/leetcode记录-1、88、20/index.html","21aa7060cfc348a955692d39e064fe42"],["E:/GitHubBlog/public/2020/03/20/温度转换程序/index.html","fec7b243b7ae0d07112e99e25ae3d40e"],["E:/GitHubBlog/public/2020/03/22/数据结构第五周-递归/index.html","76e5c6407881cbea784d0e419aec1e80"],["E:/GitHubBlog/public/2020/03/22/数据结构第六周-串与数组/index.html","65ea23cec291a9c7db212838f06cbf08"],["E:/GitHubBlog/public/2020/04/02/leetcode记录-52/index.html","c19dea7b023858357091703e080db868"],["E:/GitHubBlog/public/2020/04/17/Android第八周作业-每天第一次打开应用更新数据/index.html","8b74d3ed38cca67572ab7cc57b2dc6df"],["E:/GitHubBlog/public/2020/05/21/attention模型/index.html","c408966ea95745e62384fe1a166f6e1c"],["E:/GitHubBlog/public/2020/05/21/编码器与解码器框架/index.html","5e6ef6a21128f0180d44268f00ca538e"],["E:/GitHubBlog/public/2020/07/11/思考，快与慢/index.html","c80cb615636b7dd3a43e848be1923bb3"],["E:/GitHubBlog/public/2020/07/13/20200713-0717总结/index.html","8cb680f7768246291955b18d4e5ee073"],["E:/GitHubBlog/public/2020/07/13/ZeroShotCeres Zero-Shot Relation Extraction/index.html","b57d86d11b5ed46b5b3e63cb4e3570a9"],["E:/GitHubBlog/public/2020/07/20/Keyphrase Generation A Text Summarization Struggle/index.html","33a3ad229c7d664f601fe74e04d3250c"],["E:/GitHubBlog/public/2020/07/20/TPR/index.html","c2d7bf8c5a2d3b706fbfd581f57364a4"],["E:/GitHubBlog/public/2020/07/23/Keyphrase Generation for Scientific Document Retrieval/index.html","1ca5eeb5be446c3853de3b57e98e2d34"],["E:/GitHubBlog/public/2020/07/24/20200718-0724总结/index.html","ef4e39ffbc53bc6eecdd48dd18845187"],["E:/GitHubBlog/public/2020/07/31/20200725-0731总结/index.html","deb394491dfd2621dd5790579a793aa3"],["E:/GitHubBlog/public/2020/08/07/20200801-0807总结/index.html","a14a33863e7bbe0e51081484d7cc352a"],["E:/GitHubBlog/public/2020/08/12/逻辑回归算法/index.html","e005c813ee7e4353365b29f72f6fc20b"],["E:/GitHubBlog/public/2020/08/13/李宏毅-机器学习21/index.html","03519faf046c7689bf3c12ed7d051b4c"],["E:/GitHubBlog/public/2020/08/14/20200808-0815总结/index.html","19ae2df9467ef5d8c02dad096504045e"],["E:/GitHubBlog/public/2020/09/03/20200816-0903总结/index.html","49e098f441032aee4e1b6f23d9f1c0ab"],["E:/GitHubBlog/public/2020/09/16/20200914-0918总结/index.html","f0b362a55476fe8caaac5fde2926c3d3"],["E:/GitHubBlog/public/2020/09/25/20200919-0925总结/index.html","b0fc1124fc87de3d39e573d10e625dc2"],["E:/GitHubBlog/public/2020/10/06/《Deep Keyphrase Generation》详细解析！/index.html","fa72fb876e6706fbc21fca9f9e46d112"],["E:/GitHubBlog/public/2020/10/16/20201008-1017总结/index.html","a76b0ec8c116eee868a41473676c6789"],["E:/GitHubBlog/public/2020/10/20/特征选择/index.html","35529ddd583838900c7bb2cae8a77ce2"],["E:/GitHubBlog/public/2020/10/21/PCA/index.html","780b8b34933585ee637f6bd3da5335ce"],["E:/GitHubBlog/public/archives/2020/01/index.html","859e5011ab59879da9dc65a48369f8d2"],["E:/GitHubBlog/public/archives/2020/02/index.html","5eec93fe1c044a5c11129fd83bbd10a6"],["E:/GitHubBlog/public/archives/2020/03/index.html","be804d0c3d8768ebd332e47d18c6e400"],["E:/GitHubBlog/public/archives/2020/04/index.html","8081fc44ce92602040b906fd46929cf7"],["E:/GitHubBlog/public/archives/2020/05/index.html","b9d4182613f4f4fd55ea227950460bc2"],["E:/GitHubBlog/public/archives/2020/07/index.html","f9101090594d5e1ab3b1827aa4557adf"],["E:/GitHubBlog/public/archives/2020/08/index.html","6a9a3ece86e5134368b431dd41c61b2b"],["E:/GitHubBlog/public/archives/2020/09/index.html","d253cb8d76800a03eea24909ca3a3b46"],["E:/GitHubBlog/public/archives/2020/10/index.html","88dab62352dd88f4dbcfc31a882b58b0"],["E:/GitHubBlog/public/archives/2020/index.html","79cfbb5060696c27cba58d141c53c383"],["E:/GitHubBlog/public/archives/2020/page/2/index.html","b09ef007b3ad4959bd6fb87d73eb07d4"],["E:/GitHubBlog/public/archives/2020/page/3/index.html","d2285745ed5972ee28677e45990e0439"],["E:/GitHubBlog/public/archives/2020/page/4/index.html","631c1caad9a0fbc535ff1eb74affa3f7"],["E:/GitHubBlog/public/archives/index.html","a97da124b3035466836dba0870518723"],["E:/GitHubBlog/public/archives/page/2/index.html","166d15927a4fe2f09c5dbe7ff8e56e47"],["E:/GitHubBlog/public/archives/page/3/index.html","8ee9e820d21f010cad66e639fd8ee3b5"],["E:/GitHubBlog/public/archives/page/4/index.html","5363dde3b2f48648777f95b7778aa979"],["E:/GitHubBlog/public/css/404.css","b1bb50e1fabe41adcec483f6427fb3aa"],["E:/GitHubBlog/public/css/index.css","2389364df340948a5b822422956c2c87"],["E:/GitHubBlog/public/css/var.css","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/img/algolia.svg","fd40b88ac5370a5353a50b8175c1f367"],["E:/GitHubBlog/public/img/avatar.png","6cc4a809d23e3d8946a299ae4ce4e4cd"],["E:/GitHubBlog/public/index.html","b17bc8479bbabb8ed59280dbe750935a"],["E:/GitHubBlog/public/js/copy.js","45aae54bf2e43ac27ecc41eb5e0bacf7"],["E:/GitHubBlog/public/js/fancybox.js","f84d626654b9bbc05720952b3effe062"],["E:/GitHubBlog/public/js/fireworks.js","35933ce61c158ef9c33b5e089fe757ac"],["E:/GitHubBlog/public/js/head.js","347edd99f8e3921b45fa617e839d8182"],["E:/GitHubBlog/public/js/hexo-theme-melody.js","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/js/katex.js","d971ba8b8dee1120ef66744b89cfd2b1"],["E:/GitHubBlog/public/js/scroll.js","a12ad7e37b9e325f8da3d0523d3757c7"],["E:/GitHubBlog/public/js/search/algolia.js","53160985d32d6fd66d98aa0e05b081ac"],["E:/GitHubBlog/public/js/search/local-search.js","1565b508bd866ed6fbd724a3858af5d8"],["E:/GitHubBlog/public/js/sidebar.js","d24db780974e661198eeb2e45f20a28f"],["E:/GitHubBlog/public/js/third-party/anime.min.js","9b4bbe6deb700e1c3606eab732f5eea5"],["E:/GitHubBlog/public/js/third-party/canvas-ribbon.js","09c181544ddff1db701db02ac30453e0"],["E:/GitHubBlog/public/js/third-party/jquery.fancybox.min.js","3c9fa1c1199cd4f874d855ecb1641335"],["E:/GitHubBlog/public/js/third-party/jquery.min.js","c9f5aeeca3ad37bf2aa006139b935f0a"],["E:/GitHubBlog/public/js/third-party/reveal/head.min.js","aad121203010122e05f1766d92385214"],["E:/GitHubBlog/public/js/third-party/reveal/highlight.min.js","44594243bec43813a16371af8fe7e105"],["E:/GitHubBlog/public/js/third-party/reveal/markdown.min.js","7ec4cef5a7fe3f0bf0eb4dc6d7bca114"],["E:/GitHubBlog/public/js/third-party/reveal/marked.min.js","c2a88705e206d71dc21fdc4445349127"],["E:/GitHubBlog/public/js/third-party/reveal/math.min.js","0a278fee2e57c530ab07f7d2d9ea8d96"],["E:/GitHubBlog/public/js/third-party/reveal/notes.min.js","89a0dfae4d706f9c75b317f686c3aa14"],["E:/GitHubBlog/public/js/third-party/reveal/reveal.min.js","8988419d67efb5fe93e291a357e26ec9"],["E:/GitHubBlog/public/js/third-party/reveal/zoom.min.js","9791f96e63e7d534cba2b67d4bda0419"],["E:/GitHubBlog/public/js/third-party/velocity.min.js","64da069aba987ea0512cf610600a56d1"],["E:/GitHubBlog/public/js/third-party/velocity.ui.min.js","c8ca438424a080620f7b2f4ee4b0fff1"],["E:/GitHubBlog/public/js/transition.js","911db4268f0f6621073afcced9e1bfef"],["E:/GitHubBlog/public/js/utils.js","3ff3423d966a1c351e9867813b3f6d36"],["E:/GitHubBlog/public/page/2/index.html","a2d0c4d0cb4e32d7a4b841333060ad10"],["E:/GitHubBlog/public/page/3/index.html","3cae97cb8b0d7e0989831931f99b7268"],["E:/GitHubBlog/public/page/4/index.html","5db62a4b08aca5d2a9bd2495483f69a1"],["E:/GitHubBlog/public/tags/Android/index.html","577ccf38dd632041a7eff08f9080ff79"],["E:/GitHubBlog/public/tags/NLP/index.html","6777a436214e25157b84744ac646df44"],["E:/GitHubBlog/public/tags/NLP/page/2/index.html","9c630afdee5d29b67d4ed1b822cc5ba3"],["E:/GitHubBlog/public/tags/R/index.html","c47ed96eb7d5b47b8ec527d938ebf5e4"],["E:/GitHubBlog/public/tags/index.html","d7d7ae7d8c79589f687313d8dd989872"],["E:/GitHubBlog/public/tags/java/index.html","cc49b3ff1075e8f3f2e9518ef75961f0"],["E:/GitHubBlog/public/tags/leetcode/index.html","d4b1794d93deb342551bdebc24a39d11"],["E:/GitHubBlog/public/tags/python/index.html","032942961a2c801526b8680511cb5e97"],["E:/GitHubBlog/public/tags/总结/index.html","7232895f924864cb8e76647076715747"],["E:/GitHubBlog/public/tags/数据分析/index.html","aa901a8e3f7db5931ea6cc35aab91e1d"],["E:/GitHubBlog/public/tags/数据挖掘/index.html","542e91b2ad44b0dce5986505ba556b18"],["E:/GitHubBlog/public/tags/数据结构/index.html","b953901b3b079bcc1886dca30fc1d6df"],["E:/GitHubBlog/public/tags/机器学习/index.html","fe7eeb8e3f05009d938e9a0a84a5d551"],["E:/GitHubBlog/public/tags/深度学习/index.html","9eb88dbb2c4531b29f22ccdab32dd066"],["E:/GitHubBlog/public/tags/爬虫/index.html","36850d8c6abeb52edd1fa39777fd7a00"],["E:/GitHubBlog/public/tags/笔记/index.html","f3f4cc9d109ba56b10821f46a6fbb03e"],["E:/GitHubBlog/public/tags/论文/index.html","429344d2ae16557a9f0b09bc514e108f"],["E:/GitHubBlog/public/tags/论文/page/2/index.html","093d2842a62ec7da5fe9012855e403df"],["E:/GitHubBlog/public/tags/读书笔记/index.html","6a42a7e81b50321ebfd5af759fcc2613"]];
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







