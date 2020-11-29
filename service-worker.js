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

var precacheConfig = [["E:/GitHubBlog/public/2020/01/28/hello-world/index.html","e44dc1b911407ccffafce9299dc8375d"],["E:/GitHubBlog/public/2020/02/08/python爬虫学习总结/index.html","77ab104a80c7163d4d123930e4a38e9a"],["E:/GitHubBlog/public/2020/02/19/数据结构第一周-绪论/index.html","57e45aa8c7fe8bd96557eca8f8eb8b74"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-1/index.html","f7b6477a163e2049b4795098f415aa28"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-2-创建数据集/index.html","44ded06c021baaf1c4b98599c7e90148"],["E:/GitHubBlog/public/2020/03/01/《R语言实战》学习总结-3-图形初阶/index.html","d450e30d8bd0ab4c442883e41adeec53"],["E:/GitHubBlog/public/2020/03/02/数据结构第二周-线性结构1/index.html","6ac877f33f31d8e4f7b031278f137122"],["E:/GitHubBlog/public/2020/03/03/《R语言实战》学习总结-4-基本数据管理/index.html","9dfdb7390c0f9b2f6a6a3975d5bd4e44"],["E:/GitHubBlog/public/2020/03/14/数据结构第三周-线性结构2/index.html","a973057761bb82ca528922bcb83dab28"],["E:/GitHubBlog/public/2020/03/14/数据结构第四周-栈与队列/index.html","5fe19d935505156bbbaaa2f3d0432c84"],["E:/GitHubBlog/public/2020/03/17/leetcode记录-1、88、20/index.html","7a24271a5eaed5619f5c8ae8cb11ead9"],["E:/GitHubBlog/public/2020/03/20/温度转换程序/index.html","180086ee1f915e5a5999e7d78c825c5b"],["E:/GitHubBlog/public/2020/03/22/数据结构第五周-递归/index.html","24e8c54717481054f60b8aa0766ee6bb"],["E:/GitHubBlog/public/2020/03/22/数据结构第六周-串与数组/index.html","38c0551df7dc3489543f122c85731bba"],["E:/GitHubBlog/public/2020/04/02/leetcode记录-52/index.html","4ac15f775904a4d779ae0288def0f826"],["E:/GitHubBlog/public/2020/04/17/Android第八周作业-每天第一次打开应用更新数据/index.html","89641647e02f75e714a4a6b592dcd20b"],["E:/GitHubBlog/public/2020/05/21/attention模型/index.html","fb0346635f099e526de9b2481da9fb8a"],["E:/GitHubBlog/public/2020/05/21/编码器与解码器框架/index.html","d8f8732a8904955030b6837e2be5f96b"],["E:/GitHubBlog/public/2020/07/11/思考，快与慢/index.html","48ef8108e12536717bc31035c266ffb5"],["E:/GitHubBlog/public/2020/07/13/20200713-0717总结/index.html","659b05f39117896677b3d8d224c8e41c"],["E:/GitHubBlog/public/2020/07/13/ZeroShotCeres Zero-Shot Relation Extraction/index.html","cf83917da5c8304b1c8ea590c043a8d5"],["E:/GitHubBlog/public/2020/07/20/Keyphrase Generation A Text Summarization Struggle/index.html","d3992d8d6f4cb15b48b48dbb1c5be372"],["E:/GitHubBlog/public/2020/07/20/TPR/index.html","7c25484f2552b2490ad18b5a1807a4c4"],["E:/GitHubBlog/public/2020/07/23/Keyphrase Generation for Scientific Document Retrieval/index.html","046b6c404613f0f6a0bea68f44c02215"],["E:/GitHubBlog/public/2020/07/24/20200718-0724总结/index.html","e13def2001198a03efdcb202612c2008"],["E:/GitHubBlog/public/2020/07/31/20200725-0731总结/index.html","116504d50d8693cbe486b93b2d9c762d"],["E:/GitHubBlog/public/2020/08/07/20200801-0807总结/index.html","4655cd603d574d43970b432c3fdf9841"],["E:/GitHubBlog/public/2020/08/12/逻辑回归算法/index.html","55d532cf6a2f8796437519ce7d3add94"],["E:/GitHubBlog/public/2020/08/13/李宏毅-机器学习21/index.html","768d234293c93c9cffe1d684b4fbabe7"],["E:/GitHubBlog/public/2020/08/14/20200808-0815总结/index.html","a66c114f51ec8293c9c66983709a84a2"],["E:/GitHubBlog/public/2020/09/03/20200816-0903总结/index.html","a4273d9ec7ff8666c3f814fd97e48a9c"],["E:/GitHubBlog/public/2020/09/16/20200914-0918总结/index.html","5df46843375ca577cbcdcd8481a78484"],["E:/GitHubBlog/public/2020/09/25/20200919-0925总结/index.html","93c9b6ae33ea840e83260f1b57fad175"],["E:/GitHubBlog/public/2020/10/06/《Deep Keyphrase Generation》详细解析！/index.html","179b9926bfac51d78f37d8cec0f6d559"],["E:/GitHubBlog/public/2020/10/16/20201008-1017总结/index.html","88c181ab9c89f05eee6e4633befee742"],["E:/GitHubBlog/public/2020/10/20/特征选择的几种方法(一)————Filter过滤法/index.html","5147c8ef7bde4df1ae189a8a9fdb36b0"],["E:/GitHubBlog/public/2020/10/21/PCA/index.html","099e3fc272a65fcf258d142ab52a5482"],["E:/GitHubBlog/public/2020/10/22/20201018-1023总结/index.html","aab9db465d169310f0bc11d67a7836c7"],["E:/GitHubBlog/public/2020/10/24/20201024-1030/index.html","c7caa319eae3b28592a09d09861fb420"],["E:/GitHubBlog/public/2020/10/30/1030讨论记录/index.html","1e6ad72377ebee14ba5d680792453827"],["E:/GitHubBlog/public/2020/11/03/20201103-总结/index.html","0d3e32ed6c666f0be77ea3eef36b63ac"],["E:/GitHubBlog/public/2020/11/05/线性回归与逻辑回归/index.html","41d721b5a104aa4f1e22e57fe1195e31"],["E:/GitHubBlog/public/2020/11/07/决策树/index.html","9d62a82b1c737b1e147d15ca3a548863"],["E:/GitHubBlog/public/2020/11/13/20201107-1113总结/index.html","23ba154d631849d9251224661230e5cc"],["E:/GitHubBlog/public/2020/11/15/20201114-119总结/index.html","890c40e09681bfa50e6279a80da0b383"],["E:/GitHubBlog/public/2020/11/16/手推一个SVM/index.html","b51a68b9b915dcc3a67ae65830f3c5d4"],["E:/GitHubBlog/public/2020/11/27/20201127左右的总结/index.html","2b0122b04816820de4394e4a6d9f066f"],["E:/GitHubBlog/public/archives/2020/01/index.html","42a2455efba5c6bdc24f6075568050e7"],["E:/GitHubBlog/public/archives/2020/02/index.html","9cef1ecba54ed26114dd4f5c9643400d"],["E:/GitHubBlog/public/archives/2020/03/index.html","70d4ad76394af9bf50136c8f01552dc5"],["E:/GitHubBlog/public/archives/2020/04/index.html","db40b3bebee2ae671f16d630ecfe4cac"],["E:/GitHubBlog/public/archives/2020/05/index.html","b30cc29e20fa7fb23c6684342b80d625"],["E:/GitHubBlog/public/archives/2020/07/index.html","fda7f8e3c77017a610dc412b12e13940"],["E:/GitHubBlog/public/archives/2020/08/index.html","7e812ca8909d0b9db89a4765801fc592"],["E:/GitHubBlog/public/archives/2020/09/index.html","92a988fa4ec52f0a5d7b93705865a22c"],["E:/GitHubBlog/public/archives/2020/10/index.html","86b857584e21ce4090a64f664f0f7012"],["E:/GitHubBlog/public/archives/2020/11/index.html","0dfa2d5a096ac5807e9326f5f2563d1a"],["E:/GitHubBlog/public/archives/2020/index.html","eaecc1aaac29970f5b600af4dd898cf0"],["E:/GitHubBlog/public/archives/2020/page/2/index.html","9105c53c25bc8006dbca6d5d52d6ecba"],["E:/GitHubBlog/public/archives/2020/page/3/index.html","e154804cd0cd71ad0c6e911ada3e1240"],["E:/GitHubBlog/public/archives/2020/page/4/index.html","e64e2fcdf627b8da4a59e9d928f7cda8"],["E:/GitHubBlog/public/archives/2020/page/5/index.html","08803c587720f9f077ae3715f59b66de"],["E:/GitHubBlog/public/archives/index.html","20111dba34cfa7db7028faa0a181d46b"],["E:/GitHubBlog/public/archives/page/2/index.html","542d3eef5a6bfd17520303fcb6d36a38"],["E:/GitHubBlog/public/archives/page/3/index.html","195f0ee86c72aa2283f62539fca0776c"],["E:/GitHubBlog/public/archives/page/4/index.html","6af74403ce7254851872638a830f64ca"],["E:/GitHubBlog/public/archives/page/5/index.html","7537db5b4a824c578546060e01951261"],["E:/GitHubBlog/public/assets/css/APlayer.min.css","fbe994054426fadb2dff69d824c5c67a"],["E:/GitHubBlog/public/assets/js/APlayer.min.js","8f1017e7a73737e631ff95fa51e4e7d7"],["E:/GitHubBlog/public/assets/js/Meting.min.js","bfac0368480fd344282ec018d28f173d"],["E:/GitHubBlog/public/css/404.css","b1bb50e1fabe41adcec483f6427fb3aa"],["E:/GitHubBlog/public/css/index.css","2389364df340948a5b822422956c2c87"],["E:/GitHubBlog/public/css/var.css","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/img/algolia.svg","fd40b88ac5370a5353a50b8175c1f367"],["E:/GitHubBlog/public/img/avatar.png","6cc4a809d23e3d8946a299ae4ce4e4cd"],["E:/GitHubBlog/public/index.html","290d75b6cae465c4ce50b9e3c30f7ef0"],["E:/GitHubBlog/public/js/copy.js","45aae54bf2e43ac27ecc41eb5e0bacf7"],["E:/GitHubBlog/public/js/fancybox.js","f84d626654b9bbc05720952b3effe062"],["E:/GitHubBlog/public/js/fireworks.js","35933ce61c158ef9c33b5e089fe757ac"],["E:/GitHubBlog/public/js/head.js","347edd99f8e3921b45fa617e839d8182"],["E:/GitHubBlog/public/js/hexo-theme-melody.js","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/js/katex.js","d971ba8b8dee1120ef66744b89cfd2b1"],["E:/GitHubBlog/public/js/scroll.js","a12ad7e37b9e325f8da3d0523d3757c7"],["E:/GitHubBlog/public/js/search/algolia.js","53160985d32d6fd66d98aa0e05b081ac"],["E:/GitHubBlog/public/js/search/local-search.js","1565b508bd866ed6fbd724a3858af5d8"],["E:/GitHubBlog/public/js/sidebar.js","d24db780974e661198eeb2e45f20a28f"],["E:/GitHubBlog/public/js/third-party/anime.min.js","9b4bbe6deb700e1c3606eab732f5eea5"],["E:/GitHubBlog/public/js/third-party/canvas-ribbon.js","09c181544ddff1db701db02ac30453e0"],["E:/GitHubBlog/public/js/third-party/jquery.fancybox.min.js","3c9fa1c1199cd4f874d855ecb1641335"],["E:/GitHubBlog/public/js/third-party/jquery.min.js","c9f5aeeca3ad37bf2aa006139b935f0a"],["E:/GitHubBlog/public/js/third-party/reveal/head.min.js","aad121203010122e05f1766d92385214"],["E:/GitHubBlog/public/js/third-party/reveal/highlight.min.js","44594243bec43813a16371af8fe7e105"],["E:/GitHubBlog/public/js/third-party/reveal/markdown.min.js","7ec4cef5a7fe3f0bf0eb4dc6d7bca114"],["E:/GitHubBlog/public/js/third-party/reveal/marked.min.js","c2a88705e206d71dc21fdc4445349127"],["E:/GitHubBlog/public/js/third-party/reveal/math.min.js","0a278fee2e57c530ab07f7d2d9ea8d96"],["E:/GitHubBlog/public/js/third-party/reveal/notes.min.js","89a0dfae4d706f9c75b317f686c3aa14"],["E:/GitHubBlog/public/js/third-party/reveal/reveal.min.js","8988419d67efb5fe93e291a357e26ec9"],["E:/GitHubBlog/public/js/third-party/reveal/zoom.min.js","9791f96e63e7d534cba2b67d4bda0419"],["E:/GitHubBlog/public/js/third-party/velocity.min.js","64da069aba987ea0512cf610600a56d1"],["E:/GitHubBlog/public/js/third-party/velocity.ui.min.js","c8ca438424a080620f7b2f4ee4b0fff1"],["E:/GitHubBlog/public/js/transition.js","911db4268f0f6621073afcced9e1bfef"],["E:/GitHubBlog/public/js/utils.js","3ff3423d966a1c351e9867813b3f6d36"],["E:/GitHubBlog/public/page/2/index.html","d06be206d90609167782b3bd980bf6ad"],["E:/GitHubBlog/public/page/3/index.html","b1fe3fc6cfa7c02f2c60819f730346cf"],["E:/GitHubBlog/public/page/4/index.html","377c5466e062a095e1b6fa39f1cbdce3"],["E:/GitHubBlog/public/page/5/index.html","85819449e94b92a01ff1476089dd370b"],["E:/GitHubBlog/public/tags/Android/index.html","688f845e62d056318fcff6c3f4ea00ec"],["E:/GitHubBlog/public/tags/NLP/index.html","82cbf3451830fa3f3a1f42d25ca5a6f4"],["E:/GitHubBlog/public/tags/NLP/page/2/index.html","710a03f172f30a988d718e7718565655"],["E:/GitHubBlog/public/tags/R/index.html","430c22ea3044dd7de545082d8a860300"],["E:/GitHubBlog/public/tags/index.html","0a14dbcb825a22193eb310f42b53ac07"],["E:/GitHubBlog/public/tags/java/index.html","513396a1d16e863e9e493aae5325d865"],["E:/GitHubBlog/public/tags/leetcode/index.html","1cdefccc553ab47a24669072279f3e04"],["E:/GitHubBlog/public/tags/python/index.html","31e6f5ae4229bbb4855f1410699b4812"],["E:/GitHubBlog/public/tags/总结/index.html","9affe2b0fa21a0a188b6d30ed5797015"],["E:/GitHubBlog/public/tags/总结/page/2/index.html","5fea251ca837fcd65530c991d70b050a"],["E:/GitHubBlog/public/tags/数据分析/index.html","6633ad3fc2239efe3597c136f536f174"],["E:/GitHubBlog/public/tags/数据挖掘/index.html","0cd467720f2b0a22fb313652fe513f20"],["E:/GitHubBlog/public/tags/数据结构/index.html","2567ed8cb2ba1f67c1e3aaad1120a770"],["E:/GitHubBlog/public/tags/机器学习/index.html","8156e9850b967f6bf24f429cec6564bc"],["E:/GitHubBlog/public/tags/深度学习/index.html","5d411ccff8c43ef5b64ea4e8e04c364f"],["E:/GitHubBlog/public/tags/爬虫/index.html","dd48a637e4e714a7d127c5c9d722f098"],["E:/GitHubBlog/public/tags/笔记/index.html","8d5a74e80fa1faf3041994c772a307dc"],["E:/GitHubBlog/public/tags/论文/index.html","1f6275aab951ff45bc28208a065c3393"],["E:/GitHubBlog/public/tags/论文/page/2/index.html","de694f8137b561d9e45aa0a6e29d13d7"],["E:/GitHubBlog/public/tags/论文/page/3/index.html","d1d14af431b37314653530d1eec01991"],["E:/GitHubBlog/public/tags/读书笔记/index.html","405524da30cc36fea9301b01aff290ae"]];
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







