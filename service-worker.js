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

var precacheConfig = [["E:/GitHubBlog/public/2020/01/28/hello-world/index.html","4256bc68115705ceba481a3991b5fa2e"],["E:/GitHubBlog/public/2020/02/08/python爬虫学习总结/index.html","1d71adc1dccd62c92f39f6df5e5e38eb"],["E:/GitHubBlog/public/2020/02/19/数据结构第一周-绪论/index.html","e1364f0d058422b094b05df528ed005f"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-1/index.html","eaac4b7b72f0572bb9dbbb1a014fa849"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-2-创建数据集/index.html","d039ca0f685a1022508a44614e9b4423"],["E:/GitHubBlog/public/2020/03/01/《R语言实战》学习总结-3-图形初阶/index.html","f28fc9f3e2510bc0a040197c52643ee8"],["E:/GitHubBlog/public/2020/03/02/数据结构第二周-线性结构1/index.html","17f5f142c6f64bf61ed4e1aed57d596c"],["E:/GitHubBlog/public/2020/03/03/《R语言实战》学习总结-4-基本数据管理/index.html","56c520fc3cf805c80fb7077c45f7b75c"],["E:/GitHubBlog/public/2020/03/14/数据结构第三周-线性结构2/index.html","b22bcd5d2310d6127dce9347aa9f6f03"],["E:/GitHubBlog/public/2020/03/14/数据结构第四周-栈与队列/index.html","02f102782004d97c5ca4d1daf8b34603"],["E:/GitHubBlog/public/2020/03/17/leetcode记录-1、88、20/index.html","b9f5285987a76250dd154c8c5fbd5566"],["E:/GitHubBlog/public/2020/03/20/温度转换程序/index.html","46a8e32c97a99e99db3be1357abe6499"],["E:/GitHubBlog/public/2020/03/22/数据结构第五周-递归/index.html","d8cf0df305c8cf2d7e185186553a0683"],["E:/GitHubBlog/public/2020/03/22/数据结构第六周-串与数组/index.html","e5af54317162da02ea03043100fedec6"],["E:/GitHubBlog/public/2020/04/02/leetcode记录-52/index.html","983901ace6ea4a6289015ca4de809302"],["E:/GitHubBlog/public/2020/04/17/Android第八周作业-每天第一次打开应用更新数据/index.html","896c746b9321ee210977f90641619a52"],["E:/GitHubBlog/public/2020/05/21/attention模型/index.html","384af3742d96ac8af4917a0591d3fbb2"],["E:/GitHubBlog/public/2020/05/21/编码器与解码器框架/index.html","a5bff51889e2e12becc7f4010e4c2e8f"],["E:/GitHubBlog/public/2020/07/11/思考，快与慢/index.html","bd5deee33084406cda6c0dc7c013e679"],["E:/GitHubBlog/public/2020/07/13/20200713-0717总结/index.html","8b188896f84471d9794588d964afcd6e"],["E:/GitHubBlog/public/2020/07/13/ZeroShotCeres Zero-Shot Relation Extraction/index.html","80a2e08ccc07eb59b9f0b17a9f0783f3"],["E:/GitHubBlog/public/2020/07/20/Keyphrase Generation A Text Summarization Struggle/index.html","b0b20895f955a704475e354bed80dd99"],["E:/GitHubBlog/public/2020/07/20/TPR/index.html","9233bba86fd4d53dff8812ae510312d9"],["E:/GitHubBlog/public/2020/07/23/Keyphrase Generation for Scientific Document Retrieval/index.html","a1ae4fca10d72160228badfbef1040aa"],["E:/GitHubBlog/public/2020/07/24/20200718-0724总结/index.html","b01dfe61d0e04f388daad17f09a11fb4"],["E:/GitHubBlog/public/2020/07/31/20200725-0731总结/index.html","e16e4be26ef6eeee000c98fdec7e015c"],["E:/GitHubBlog/public/2020/08/07/20200801-0807总结/index.html","85fcc8639acf6ecd99a790d5961dafa2"],["E:/GitHubBlog/public/2020/08/12/逻辑回归算法/index.html","d6f905469ecc085be25a15b11887f047"],["E:/GitHubBlog/public/2020/08/13/李宏毅-机器学习21/index.html","ca42367dc6b6b22a5ec3c3fc6b60b40a"],["E:/GitHubBlog/public/2020/08/14/20200808-0815总结/index.html","8b571aa39ecda3b7a5159c3072402ff3"],["E:/GitHubBlog/public/2020/09/03/20200816-0903总结/index.html","4a5b9a9a799e8f77f8b3fa5ef7c7927a"],["E:/GitHubBlog/public/2020/09/16/20200914-0918总结/index.html","26f73dca172705bf080c26af8905c17d"],["E:/GitHubBlog/public/2020/09/25/20200919-0925总结/index.html","313f1f46f420d4496a13f9b62f5d9d77"],["E:/GitHubBlog/public/2020/10/06/《Deep Keyphrase Generation》详细解析！/index.html","cf83b2ee23b903540dce6dcc9a24b0ee"],["E:/GitHubBlog/public/2020/10/16/20201008-1017总结/index.html","fe0fcba46d2a48a025341360967e97f7"],["E:/GitHubBlog/public/2020/10/20/特征选择的几种方法(一)————Filter过滤法/index.html","bcfb2ed17817ae5b2c5e2d0e28f5e9f4"],["E:/GitHubBlog/public/2020/10/21/PCA/index.html","ef52352a9db3b9ca2d32ec1c0a3d3a5d"],["E:/GitHubBlog/public/2020/10/22/20201018-1023总结/index.html","00f0c11bfa2cb44427fa79998b7545ef"],["E:/GitHubBlog/public/2020/10/24/20201024-1030/index.html","c0dde8262b5c0d7ac29799ccb1201273"],["E:/GitHubBlog/public/2020/10/30/1030讨论记录/index.html","d26c4478fb9de4e4ca1e0e3ce288a411"],["E:/GitHubBlog/public/2020/11/02/《如何写一个商业计划书》读后感/index.html","c7105917aaa3d0d0336a6ce877a14611"],["E:/GitHubBlog/public/2020/11/03/20201103-总结/index.html","459216e28b2e1e203b971c81ff46829f"],["E:/GitHubBlog/public/2020/11/04/《思考，快与慢》读后感/index.html","ce994095881341959ff9bb522638fd1e"],["E:/GitHubBlog/public/2020/11/05/优化方法!!——一些前置知识/index.html","ce215e34dd2799ac171034cb6545201d"],["E:/GitHubBlog/public/2020/11/07/决策树/index.html","df98b63e11446e020866d3c8387a08a7"],["E:/GitHubBlog/public/2020/11/13/20201107-1113总结/index.html","d66c84e32f5acf2d06afe69a55a2843a"],["E:/GitHubBlog/public/2020/11/15/20201114-119总结/index.html","39c44e2677ea68c316e260d16805f96b"],["E:/GitHubBlog/public/2020/11/15/手推一个SVM/index.html","f697078623d2c83b4118b18c3e2da178"],["E:/GitHubBlog/public/archives/2020/01/index.html","6e1de1f8963afffa04610aef6c3c5449"],["E:/GitHubBlog/public/archives/2020/02/index.html","10539ea15d42c94647177328508d1edb"],["E:/GitHubBlog/public/archives/2020/03/index.html","3d15b35b239120f679c8cc9c39274e9b"],["E:/GitHubBlog/public/archives/2020/04/index.html","1cb0e196689d40094f021f1a04d775df"],["E:/GitHubBlog/public/archives/2020/05/index.html","a057824e8169ddc10121a3d6cb81f0c2"],["E:/GitHubBlog/public/archives/2020/07/index.html","38ad9a8d1a9358a44e8a6033303ba685"],["E:/GitHubBlog/public/archives/2020/08/index.html","e86cc02ce0d380fde2bc4298fe13f4a2"],["E:/GitHubBlog/public/archives/2020/09/index.html","92eb584aaf9c64626c2d5cc8eea0e5ab"],["E:/GitHubBlog/public/archives/2020/10/index.html","2672421ec5602656fb83aaa4da6faa26"],["E:/GitHubBlog/public/archives/2020/11/index.html","f0eee2faec64b71f0dd3f9c1f3398d3a"],["E:/GitHubBlog/public/archives/2020/index.html","3608fa38f03ccfefac7f52817355b57d"],["E:/GitHubBlog/public/archives/2020/page/2/index.html","3144c23ad9f29151bf38b5be5b9f2af4"],["E:/GitHubBlog/public/archives/2020/page/3/index.html","68e7c959800fe127d8ff2d610f234dfa"],["E:/GitHubBlog/public/archives/2020/page/4/index.html","d9c30425ec68d9afe52b81b286361be1"],["E:/GitHubBlog/public/archives/2020/page/5/index.html","fbcefbe6a0199ea1df9de01e73f59924"],["E:/GitHubBlog/public/archives/index.html","7e1c245791456bbf6c702c40bf9e9298"],["E:/GitHubBlog/public/archives/page/2/index.html","a07de20255f4ba63cd708f13d0667e7e"],["E:/GitHubBlog/public/archives/page/3/index.html","4104e81ee327f9f519163c5a01ac21b5"],["E:/GitHubBlog/public/archives/page/4/index.html","be55363b994b4f8fb7ebd4cdd1b34a9e"],["E:/GitHubBlog/public/archives/page/5/index.html","3e1af07c4565861d62da4886e9b7f4f0"],["E:/GitHubBlog/public/css/404.css","b1bb50e1fabe41adcec483f6427fb3aa"],["E:/GitHubBlog/public/css/index.css","2389364df340948a5b822422956c2c87"],["E:/GitHubBlog/public/css/var.css","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/img/algolia.svg","fd40b88ac5370a5353a50b8175c1f367"],["E:/GitHubBlog/public/img/avatar.png","6cc4a809d23e3d8946a299ae4ce4e4cd"],["E:/GitHubBlog/public/index.html","ee7a66960cc603a46c941220f771d01d"],["E:/GitHubBlog/public/js/copy.js","45aae54bf2e43ac27ecc41eb5e0bacf7"],["E:/GitHubBlog/public/js/fancybox.js","f84d626654b9bbc05720952b3effe062"],["E:/GitHubBlog/public/js/fireworks.js","35933ce61c158ef9c33b5e089fe757ac"],["E:/GitHubBlog/public/js/head.js","347edd99f8e3921b45fa617e839d8182"],["E:/GitHubBlog/public/js/hexo-theme-melody.js","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/js/katex.js","d971ba8b8dee1120ef66744b89cfd2b1"],["E:/GitHubBlog/public/js/scroll.js","a12ad7e37b9e325f8da3d0523d3757c7"],["E:/GitHubBlog/public/js/search/algolia.js","53160985d32d6fd66d98aa0e05b081ac"],["E:/GitHubBlog/public/js/search/local-search.js","1565b508bd866ed6fbd724a3858af5d8"],["E:/GitHubBlog/public/js/sidebar.js","d24db780974e661198eeb2e45f20a28f"],["E:/GitHubBlog/public/js/third-party/anime.min.js","9b4bbe6deb700e1c3606eab732f5eea5"],["E:/GitHubBlog/public/js/third-party/canvas-ribbon.js","09c181544ddff1db701db02ac30453e0"],["E:/GitHubBlog/public/js/third-party/jquery.fancybox.min.js","3c9fa1c1199cd4f874d855ecb1641335"],["E:/GitHubBlog/public/js/third-party/jquery.min.js","c9f5aeeca3ad37bf2aa006139b935f0a"],["E:/GitHubBlog/public/js/third-party/reveal/head.min.js","aad121203010122e05f1766d92385214"],["E:/GitHubBlog/public/js/third-party/reveal/highlight.min.js","44594243bec43813a16371af8fe7e105"],["E:/GitHubBlog/public/js/third-party/reveal/markdown.min.js","7ec4cef5a7fe3f0bf0eb4dc6d7bca114"],["E:/GitHubBlog/public/js/third-party/reveal/marked.min.js","c2a88705e206d71dc21fdc4445349127"],["E:/GitHubBlog/public/js/third-party/reveal/math.min.js","0a278fee2e57c530ab07f7d2d9ea8d96"],["E:/GitHubBlog/public/js/third-party/reveal/notes.min.js","89a0dfae4d706f9c75b317f686c3aa14"],["E:/GitHubBlog/public/js/third-party/reveal/reveal.min.js","8988419d67efb5fe93e291a357e26ec9"],["E:/GitHubBlog/public/js/third-party/reveal/zoom.min.js","9791f96e63e7d534cba2b67d4bda0419"],["E:/GitHubBlog/public/js/third-party/velocity.min.js","64da069aba987ea0512cf610600a56d1"],["E:/GitHubBlog/public/js/third-party/velocity.ui.min.js","c8ca438424a080620f7b2f4ee4b0fff1"],["E:/GitHubBlog/public/js/transition.js","911db4268f0f6621073afcced9e1bfef"],["E:/GitHubBlog/public/js/utils.js","3ff3423d966a1c351e9867813b3f6d36"],["E:/GitHubBlog/public/page/2/index.html","06ddb3cd601bf37549b3c11707ed67fe"],["E:/GitHubBlog/public/page/3/index.html","8e2d66d35a882440110179a3fc48a6a3"],["E:/GitHubBlog/public/page/4/index.html","6d4fe5ccf20743af8b052e806da5eb82"],["E:/GitHubBlog/public/page/5/index.html","0ac0e42df65e1d502a4fc2a06c22373b"],["E:/GitHubBlog/public/tags/Android/index.html","686e6177b5b743b0615606ea0c583a12"],["E:/GitHubBlog/public/tags/NLP/index.html","ff3c2917be9ee9f8390a784baa136d8f"],["E:/GitHubBlog/public/tags/NLP/page/2/index.html","715427bea5ccf97eebe446bfd87a04a4"],["E:/GitHubBlog/public/tags/R/index.html","ba6b58b28786dd74d4263f1124802283"],["E:/GitHubBlog/public/tags/index.html","3beb403dc484b6cc332f5c28e5be59c4"],["E:/GitHubBlog/public/tags/java/index.html","185ec1d82c596733e550b961e8210ed0"],["E:/GitHubBlog/public/tags/leetcode/index.html","57a23c1306b42b85e4fc76723caa2a59"],["E:/GitHubBlog/public/tags/python/index.html","29216692b4313c8d955fa72bec716b7b"],["E:/GitHubBlog/public/tags/优化方法/index.html","aa4ba1d9f16b6cde25b4a3f28491b7bc"],["E:/GitHubBlog/public/tags/总结/index.html","e30f024de6eb02a0aaa63f31f6d1ec07"],["E:/GitHubBlog/public/tags/总结/page/2/index.html","36e758a43004b863b5661861918f2f44"],["E:/GitHubBlog/public/tags/数学/index.html","c03e2be6b004e2f88a6da2a748dde03e"],["E:/GitHubBlog/public/tags/数据分析/index.html","d976be225981c7d52e8e18ead4222751"],["E:/GitHubBlog/public/tags/数据挖掘/index.html","97016e897a34faa7ac799b86cf2870bd"],["E:/GitHubBlog/public/tags/数据结构/index.html","82223643adffa846e209e7d6dd8d926b"],["E:/GitHubBlog/public/tags/机器学习/index.html","7ea2d7669a208cad54a9c1f21a29e51f"],["E:/GitHubBlog/public/tags/深度学习/index.html","f0c5456e6db19ab8b8a6e758d382a407"],["E:/GitHubBlog/public/tags/爬虫/index.html","4634df15828cf32d6ce10d129d75e191"],["E:/GitHubBlog/public/tags/笔记/index.html","997eb675b6eb670421595a356994e749"],["E:/GitHubBlog/public/tags/论文/index.html","e2cb9aa8f003e96af84876773c820e66"],["E:/GitHubBlog/public/tags/论文/page/2/index.html","70c91d8dfedba57f537c7c6a17e395c5"],["E:/GitHubBlog/public/tags/论文/page/3/index.html","5381bcb3b387dc1764fe1e8987eb92d2"],["E:/GitHubBlog/public/tags/读书笔记/index.html","2a03a0faaad1ca4d2cf58b573a414721"]];
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







