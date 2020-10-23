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

var precacheConfig = [["E:/GitHubBlog/public/2020/01/28/hello-world/index.html","dfc048c4169a2ff8112cf89b4222c4a4"],["E:/GitHubBlog/public/2020/02/08/python爬虫学习总结/index.html","ff4eb13ea439e2f23bcafb68b279ebe9"],["E:/GitHubBlog/public/2020/02/19/数据结构第一周-绪论/index.html","fa36821fb334d6d95be89ba2cafaccc0"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-1/index.html","de38b300c9046683f1b068270aace28c"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-2-创建数据集/index.html","91a6fe9bb5539adcf3fd23e4917e1ba6"],["E:/GitHubBlog/public/2020/03/01/《R语言实战》学习总结-3-图形初阶/index.html","048fcab62a5d8c91dd2f654b9fcabe00"],["E:/GitHubBlog/public/2020/03/02/数据结构第二周-线性结构1/index.html","4c2e020074d93caaae814fdcc4f70386"],["E:/GitHubBlog/public/2020/03/03/《R语言实战》学习总结-4-基本数据管理/index.html","d4e2a7c999fb8f8ed9affb5edb33ee26"],["E:/GitHubBlog/public/2020/03/14/数据结构第三周-线性结构2/index.html","81c9bcc7059d90294268840c71026bcf"],["E:/GitHubBlog/public/2020/03/14/数据结构第四周-栈与队列/index.html","11acdeccf4e16c9b8fd71d12abc17d4f"],["E:/GitHubBlog/public/2020/03/17/leetcode记录-1、88、20/index.html","d2a18bdc4897da1ecb6927538018e60d"],["E:/GitHubBlog/public/2020/03/20/温度转换程序/index.html","5437a9729785e1368555b1994761ef21"],["E:/GitHubBlog/public/2020/03/22/数据结构第五周-递归/index.html","7fa3ebde1f20562cc2341481e48c0c50"],["E:/GitHubBlog/public/2020/03/22/数据结构第六周-串与数组/index.html","3df47f54872fd448dc312960f05cb6f6"],["E:/GitHubBlog/public/2020/04/02/leetcode记录-52/index.html","c220d3b426a9655a913d6976d7ee88af"],["E:/GitHubBlog/public/2020/04/17/Android第八周作业-每天第一次打开应用更新数据/index.html","5e6bed5fbf375c866a57e674b0905b58"],["E:/GitHubBlog/public/2020/05/21/attention模型/index.html","deb855756b9d430bcd63ddc1bd601d17"],["E:/GitHubBlog/public/2020/05/21/编码器与解码器框架/index.html","6436be842a4f6b70d36f005bec1bdddf"],["E:/GitHubBlog/public/2020/07/11/思考，快与慢/index.html","4d8fa8207af9e52cab21a88966f65743"],["E:/GitHubBlog/public/2020/07/13/20200713-0717总结/index.html","f3fa7ff575e548afee92d5fa8cd42874"],["E:/GitHubBlog/public/2020/07/13/ZeroShotCeres Zero-Shot Relation Extraction/index.html","63ed93ac4ad79ccc9b4de9e7404474cd"],["E:/GitHubBlog/public/2020/07/20/Keyphrase Generation A Text Summarization Struggle/index.html","2b8f1942cba3d1ed6510a31f3f95139c"],["E:/GitHubBlog/public/2020/07/20/TPR/index.html","1200996ca4e4933cec6ad07438bbb20c"],["E:/GitHubBlog/public/2020/07/23/Keyphrase Generation for Scientific Document Retrieval/index.html","d20a1e3df8d750a3b4bd4f2357158e82"],["E:/GitHubBlog/public/2020/07/24/20200718-0724总结/index.html","0bb87eef02166527d4855adef249690d"],["E:/GitHubBlog/public/2020/07/31/20200725-0731总结/index.html","4c3aeb98281566878031eb15aa7d0e6f"],["E:/GitHubBlog/public/2020/08/07/20200801-0807总结/index.html","41dd253dcc7afcf9740b4cebae3c2cc4"],["E:/GitHubBlog/public/2020/08/12/逻辑回归算法/index.html","8e26f77e5575b6d819b496b076a6ee10"],["E:/GitHubBlog/public/2020/08/13/李宏毅-机器学习21/index.html","8ff94887c3599d68127a2ec1762eef22"],["E:/GitHubBlog/public/2020/08/14/20200808-0815总结/index.html","577f82d49500617d8dc91716a80ac9cb"],["E:/GitHubBlog/public/2020/09/03/20200816-0903总结/index.html","827d23dfcf050d5d7a298921f292277c"],["E:/GitHubBlog/public/2020/09/16/20200914-0918总结/index.html","38f9e305ee31cb4198e8d59b612bb0f1"],["E:/GitHubBlog/public/2020/09/25/20200919-0925总结/index.html","6dfbc0ad9d1369cbcf685182acccfc8f"],["E:/GitHubBlog/public/2020/10/06/《Deep Keyphrase Generation》详细解析！/index.html","dc6e3323aafa61ed3c55045a1b9d687d"],["E:/GitHubBlog/public/2020/10/16/20201008-1017总结/index.html","2ca8354ae64bb25192b00e9f11d4b644"],["E:/GitHubBlog/public/2020/10/20/特征选择/index.html","21b153669ded17fe8d4f04840a54f343"],["E:/GitHubBlog/public/2020/10/21/PCA/index.html","d30194e2529d36149200efc52e61380a"],["E:/GitHubBlog/public/2020/10/22/20201018-1023总结/index.html","8aa786c01c33af6078ef08156678e3d7"],["E:/GitHubBlog/public/archives/2020/01/index.html","ab4cd793c98c34039ef77aa97fc08af3"],["E:/GitHubBlog/public/archives/2020/02/index.html","e4b2ef5ecbc622105c71dbad65daf8bf"],["E:/GitHubBlog/public/archives/2020/03/index.html","bc1df05151d817613e9abd49f86bc2fd"],["E:/GitHubBlog/public/archives/2020/04/index.html","ad6c0a4fefd33edbb7555e3f1a45e825"],["E:/GitHubBlog/public/archives/2020/05/index.html","c49af3d5ed7b591d63c94258f039c188"],["E:/GitHubBlog/public/archives/2020/07/index.html","cc5a38952c14c9342c6dbb48be3efc5f"],["E:/GitHubBlog/public/archives/2020/08/index.html","e3d587fd618bee3ca641555558b89a43"],["E:/GitHubBlog/public/archives/2020/09/index.html","8222c1177fe48ccc91d6108f18b9448a"],["E:/GitHubBlog/public/archives/2020/10/index.html","ed9eafef95d0ee0c81a7990ce93e3331"],["E:/GitHubBlog/public/archives/2020/index.html","7b4aa63b7c4dc99b1852bc6cab2a3644"],["E:/GitHubBlog/public/archives/2020/page/2/index.html","d0cbee89e54108d557105524bfa554f6"],["E:/GitHubBlog/public/archives/2020/page/3/index.html","d621c5aaaa13725010518d4fdb89ab18"],["E:/GitHubBlog/public/archives/2020/page/4/index.html","fb5a5af682f3d344520c0262425719ad"],["E:/GitHubBlog/public/archives/index.html","ec7c4e36737b695eb7a0e76db1c46ddd"],["E:/GitHubBlog/public/archives/page/2/index.html","97d1b288e31ea6dff304b0bdd8c1fcf5"],["E:/GitHubBlog/public/archives/page/3/index.html","fdc91ca9bacfb683559ecbd4f81641b0"],["E:/GitHubBlog/public/archives/page/4/index.html","5a57da57ac6b07a3d82d71649a451c3e"],["E:/GitHubBlog/public/css/404.css","b1bb50e1fabe41adcec483f6427fb3aa"],["E:/GitHubBlog/public/css/index.css","2389364df340948a5b822422956c2c87"],["E:/GitHubBlog/public/css/var.css","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/img/algolia.svg","fd40b88ac5370a5353a50b8175c1f367"],["E:/GitHubBlog/public/img/avatar.png","6cc4a809d23e3d8946a299ae4ce4e4cd"],["E:/GitHubBlog/public/index.html","f341e0306e35a398a4816b15e552829a"],["E:/GitHubBlog/public/js/copy.js","45aae54bf2e43ac27ecc41eb5e0bacf7"],["E:/GitHubBlog/public/js/fancybox.js","f84d626654b9bbc05720952b3effe062"],["E:/GitHubBlog/public/js/fireworks.js","35933ce61c158ef9c33b5e089fe757ac"],["E:/GitHubBlog/public/js/head.js","347edd99f8e3921b45fa617e839d8182"],["E:/GitHubBlog/public/js/hexo-theme-melody.js","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/js/katex.js","d971ba8b8dee1120ef66744b89cfd2b1"],["E:/GitHubBlog/public/js/scroll.js","a12ad7e37b9e325f8da3d0523d3757c7"],["E:/GitHubBlog/public/js/search/algolia.js","53160985d32d6fd66d98aa0e05b081ac"],["E:/GitHubBlog/public/js/search/local-search.js","1565b508bd866ed6fbd724a3858af5d8"],["E:/GitHubBlog/public/js/sidebar.js","d24db780974e661198eeb2e45f20a28f"],["E:/GitHubBlog/public/js/third-party/anime.min.js","9b4bbe6deb700e1c3606eab732f5eea5"],["E:/GitHubBlog/public/js/third-party/canvas-ribbon.js","09c181544ddff1db701db02ac30453e0"],["E:/GitHubBlog/public/js/third-party/jquery.fancybox.min.js","3c9fa1c1199cd4f874d855ecb1641335"],["E:/GitHubBlog/public/js/third-party/jquery.min.js","c9f5aeeca3ad37bf2aa006139b935f0a"],["E:/GitHubBlog/public/js/third-party/reveal/head.min.js","aad121203010122e05f1766d92385214"],["E:/GitHubBlog/public/js/third-party/reveal/highlight.min.js","44594243bec43813a16371af8fe7e105"],["E:/GitHubBlog/public/js/third-party/reveal/markdown.min.js","7ec4cef5a7fe3f0bf0eb4dc6d7bca114"],["E:/GitHubBlog/public/js/third-party/reveal/marked.min.js","c2a88705e206d71dc21fdc4445349127"],["E:/GitHubBlog/public/js/third-party/reveal/math.min.js","0a278fee2e57c530ab07f7d2d9ea8d96"],["E:/GitHubBlog/public/js/third-party/reveal/notes.min.js","89a0dfae4d706f9c75b317f686c3aa14"],["E:/GitHubBlog/public/js/third-party/reveal/reveal.min.js","8988419d67efb5fe93e291a357e26ec9"],["E:/GitHubBlog/public/js/third-party/reveal/zoom.min.js","9791f96e63e7d534cba2b67d4bda0419"],["E:/GitHubBlog/public/js/third-party/velocity.min.js","64da069aba987ea0512cf610600a56d1"],["E:/GitHubBlog/public/js/third-party/velocity.ui.min.js","c8ca438424a080620f7b2f4ee4b0fff1"],["E:/GitHubBlog/public/js/transition.js","911db4268f0f6621073afcced9e1bfef"],["E:/GitHubBlog/public/js/utils.js","3ff3423d966a1c351e9867813b3f6d36"],["E:/GitHubBlog/public/page/2/index.html","b91ab483a07f06abfc9c19e30a857920"],["E:/GitHubBlog/public/page/3/index.html","2f5ba2199ef47eafe735396a1237c190"],["E:/GitHubBlog/public/page/4/index.html","f3dd30f70e1c6e1f24744609ae4f5e14"],["E:/GitHubBlog/public/tags/Android/index.html","e0e889c2e19624b69205fd4956066ce6"],["E:/GitHubBlog/public/tags/NLP/index.html","dc21accae9a8e8f0c5c0b501a5728420"],["E:/GitHubBlog/public/tags/NLP/page/2/index.html","53c56c06f293d7f1cc33a14f963b712b"],["E:/GitHubBlog/public/tags/R/index.html","b23c7586183870fc739fff6f6c3f15f6"],["E:/GitHubBlog/public/tags/index.html","658c333879af291e09cab7d5cd28ce6d"],["E:/GitHubBlog/public/tags/java/index.html","bceb4b142cf5fd12032e48d93c372b9c"],["E:/GitHubBlog/public/tags/leetcode/index.html","dd63b2a24bcce2015ee599f534abf507"],["E:/GitHubBlog/public/tags/python/index.html","faf86a7f9688b07ea3a79e636c7b6ed2"],["E:/GitHubBlog/public/tags/总结/index.html","efc2f0ec7e38cf1ac2c5d7b1686411cc"],["E:/GitHubBlog/public/tags/总结/page/2/index.html","304a9f78e6ac40908f7380c81db819e7"],["E:/GitHubBlog/public/tags/数据分析/index.html","db58458fbf8531b18376220c0327ad2f"],["E:/GitHubBlog/public/tags/数据挖掘/index.html","13f303c6ed961b06658711895d491461"],["E:/GitHubBlog/public/tags/数据结构/index.html","66c206c020e7a5f3ce2179c2341e94e4"],["E:/GitHubBlog/public/tags/机器学习/index.html","91d822212c343e811452c86afbc4c111"],["E:/GitHubBlog/public/tags/深度学习/index.html","def916fa69370bc2b4ff1dabc9840481"],["E:/GitHubBlog/public/tags/爬虫/index.html","6db3f9dd2f1fd1dc77f125112d5c902f"],["E:/GitHubBlog/public/tags/笔记/index.html","eca549670e52c7d0ce664ea8a67f7dd2"],["E:/GitHubBlog/public/tags/论文/index.html","b964997096da194355228a9acca66912"],["E:/GitHubBlog/public/tags/论文/page/2/index.html","58bef79892f4f435f5fb7a2b03d1accf"],["E:/GitHubBlog/public/tags/读书笔记/index.html","5dcdfbe88277f050872e6764156e9c86"]];
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







