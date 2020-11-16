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

var precacheConfig = [["E:/GitHubBlog/public/2020/01/28/hello-world/index.html","b7e0d62f46db1457857f5832a8114d44"],["E:/GitHubBlog/public/2020/02/08/python爬虫学习总结/index.html","7b92921637a0f6f23a30f84461ef148b"],["E:/GitHubBlog/public/2020/02/19/数据结构第一周-绪论/index.html","b9ceb48fcc68ce1da2ab417633fe397f"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-1/index.html","eac519e256582db813982e32093ce7fc"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-2-创建数据集/index.html","5f542091578dc26c52c145ce4c31e52f"],["E:/GitHubBlog/public/2020/03/01/《R语言实战》学习总结-3-图形初阶/index.html","a00e0fb719d0bb9bd26f6f595795f9c4"],["E:/GitHubBlog/public/2020/03/02/数据结构第二周-线性结构1/index.html","fa50120711c8abef7837b8a457f9c1ec"],["E:/GitHubBlog/public/2020/03/03/《R语言实战》学习总结-4-基本数据管理/index.html","3964df1498f4ca828cc82b69c2c83822"],["E:/GitHubBlog/public/2020/03/14/数据结构第三周-线性结构2/index.html","66d73abdb57deb1ba4922cda29eb1d67"],["E:/GitHubBlog/public/2020/03/14/数据结构第四周-栈与队列/index.html","feac0bba6c7d6444e9b01a0d3a988db1"],["E:/GitHubBlog/public/2020/03/17/leetcode记录-1、88、20/index.html","2f5453132e564768a9fdcc80f57fd493"],["E:/GitHubBlog/public/2020/03/20/温度转换程序/index.html","1e7e403a10d53508098c30f7e5647100"],["E:/GitHubBlog/public/2020/03/22/数据结构第五周-递归/index.html","a48b9d2d96feee3bfd13068d3f173ff5"],["E:/GitHubBlog/public/2020/03/22/数据结构第六周-串与数组/index.html","f3b18c1727fbdacd191b96ec8fa36409"],["E:/GitHubBlog/public/2020/04/02/leetcode记录-52/index.html","9f35f362eada8713315ace25fb1ce89c"],["E:/GitHubBlog/public/2020/04/17/Android第八周作业-每天第一次打开应用更新数据/index.html","4c68f20db9e4a625f4f719296b38ab35"],["E:/GitHubBlog/public/2020/05/21/attention模型/index.html","f5df38f3443af5c9c0829699100682dd"],["E:/GitHubBlog/public/2020/05/21/编码器与解码器框架/index.html","bd3b03adf1b2fb6ae5d9e7bfcab22e5d"],["E:/GitHubBlog/public/2020/07/11/思考，快与慢/index.html","59b405a2e2647dacd1e28e18a1237f50"],["E:/GitHubBlog/public/2020/07/13/20200713-0717总结/index.html","5a5c499f08df6e6f573568195451324c"],["E:/GitHubBlog/public/2020/07/13/ZeroShotCeres Zero-Shot Relation Extraction/index.html","73ffc845ccef7bc02ee291db3af79ac3"],["E:/GitHubBlog/public/2020/07/20/Keyphrase Generation A Text Summarization Struggle/index.html","772fcaf096b4f4afbe1bd24c00236d8b"],["E:/GitHubBlog/public/2020/07/20/TPR/index.html","c480f9a24a3194f03a82365a060b8587"],["E:/GitHubBlog/public/2020/07/23/Keyphrase Generation for Scientific Document Retrieval/index.html","09d478f6a98a5ea54e691fea4edaf9b3"],["E:/GitHubBlog/public/2020/07/24/20200718-0724总结/index.html","6557fd9839acd17a03a18b1864ee021d"],["E:/GitHubBlog/public/2020/07/31/20200725-0731总结/index.html","140a3b75d7bb280af3b1dfbd8d90ad92"],["E:/GitHubBlog/public/2020/08/07/20200801-0807总结/index.html","37b9c53ffa1c5568d171a42d9e28d656"],["E:/GitHubBlog/public/2020/08/12/逻辑回归算法/index.html","ac027f9830972566fce889eb1dbaf32a"],["E:/GitHubBlog/public/2020/08/13/李宏毅-机器学习21/index.html","a8c7b648c34ffabf850061a00783ef60"],["E:/GitHubBlog/public/2020/08/14/20200808-0815总结/index.html","ccab324f9662bea7aa619d21789b5488"],["E:/GitHubBlog/public/2020/09/03/20200816-0903总结/index.html","ed446be704ec221802467110f26db980"],["E:/GitHubBlog/public/2020/09/16/20200914-0918总结/index.html","c1803d0fb040b1ea8e5bdefe7ff84e34"],["E:/GitHubBlog/public/2020/09/25/20200919-0925总结/index.html","1357a53b52678227da12911bf7fa2263"],["E:/GitHubBlog/public/2020/10/06/《Deep Keyphrase Generation》详细解析！/index.html","8b1862d1c3c94eaebf25d32be946a51e"],["E:/GitHubBlog/public/2020/10/16/20201008-1017总结/index.html","e67e181bb64f0bdd694167c0da88e3f3"],["E:/GitHubBlog/public/2020/10/20/特征选择的几种方法(一)————Filter过滤法/index.html","95d4f80462ca29f666655f4c5aee33b8"],["E:/GitHubBlog/public/2020/10/21/PCA/index.html","9d30061a9b925b6a5aa622189c2ca896"],["E:/GitHubBlog/public/2020/10/22/20201018-1023总结/index.html","51e8b87f79738ebaec3a547f0a0b460b"],["E:/GitHubBlog/public/2020/10/24/20201024-1030/index.html","faf60f5b84fdaf3f32c2bde1ed92d065"],["E:/GitHubBlog/public/2020/10/30/1030讨论记录/index.html","ce70dd3e49b7d6fe19f711885d1fccdb"],["E:/GitHubBlog/public/2020/11/02/《如何写一个商业计划书》读后感/index.html","3f3926c9f6be8374f727a106ee271e9e"],["E:/GitHubBlog/public/2020/11/03/20201103-总结/index.html","574a9f40b87f48ed7922adbd7ea1899c"],["E:/GitHubBlog/public/2020/11/04/《思考，快与慢》读后感/index.html","9c805855ecaa074c2f8ea5dba7d5b3aa"],["E:/GitHubBlog/public/2020/11/05/优化方法!!——一些前置知识/index.html","b1d194d22c9d8df4bf9a50f86d91472f"],["E:/GitHubBlog/public/2020/11/05/线性回归与逻辑回归/index.html","b25bb47ce6009f40bd8c8912c2e7ce37"],["E:/GitHubBlog/public/2020/11/07/决策树/index.html","3590b02bde3585131d530353863ca02e"],["E:/GitHubBlog/public/2020/11/13/20201107-1113总结/index.html","df32be485bd1fbb1cf679e535e6ca0fb"],["E:/GitHubBlog/public/2020/11/15/20201114-119总结/index.html","5ca075af4e1c033c260f32553ec8f0c9"],["E:/GitHubBlog/public/2020/11/16/手推一个SVM/index.html","d07fe6a346d27554b0cab48854d4d634"],["E:/GitHubBlog/public/archives/2020/01/index.html","138c7e65343efe6f710af23c03170983"],["E:/GitHubBlog/public/archives/2020/02/index.html","0210a9a957f1bb3c5cf9430e078ad84f"],["E:/GitHubBlog/public/archives/2020/03/index.html","aa3777c985f2bbc1b8b5878da1b15f5c"],["E:/GitHubBlog/public/archives/2020/04/index.html","154c47d9eabafcea87e26a4336975f22"],["E:/GitHubBlog/public/archives/2020/05/index.html","93bd4b10793a01877072c22db690dd70"],["E:/GitHubBlog/public/archives/2020/07/index.html","8c67d075702215783b34c67736e2d6e8"],["E:/GitHubBlog/public/archives/2020/08/index.html","6d522084e3112c8b7e7e7b776c448f64"],["E:/GitHubBlog/public/archives/2020/09/index.html","f9b81dcfa17a70096fa1ae5d7242cee7"],["E:/GitHubBlog/public/archives/2020/10/index.html","4cdcde111186b536f741874b90cca3aa"],["E:/GitHubBlog/public/archives/2020/11/index.html","0d8d3647403a68c77cd2f1801cda243c"],["E:/GitHubBlog/public/archives/2020/index.html","a2287616e076f2f75ef7414a1d780441"],["E:/GitHubBlog/public/archives/2020/page/2/index.html","989f3593ecabfa0f6fe61dc1a7fc6a6a"],["E:/GitHubBlog/public/archives/2020/page/3/index.html","5a1f652e80a4cd1e98c6f2659e2ecc4b"],["E:/GitHubBlog/public/archives/2020/page/4/index.html","d558be3e7022b7dac908a08e9bbba4a5"],["E:/GitHubBlog/public/archives/2020/page/5/index.html","95cce70220e27259a0a4c2e2ff41664c"],["E:/GitHubBlog/public/archives/index.html","3a42ec90dca757b274070e49b862f703"],["E:/GitHubBlog/public/archives/page/2/index.html","138e78478e236c91a957d16eb8e79703"],["E:/GitHubBlog/public/archives/page/3/index.html","27972b5b7ab5010776f347dd7fb731e3"],["E:/GitHubBlog/public/archives/page/4/index.html","e26090af7ab016092889478a85f58e6b"],["E:/GitHubBlog/public/archives/page/5/index.html","75e24b53bf918a4a2bc3b22cd07e47f8"],["E:/GitHubBlog/public/assets/css/APlayer.min.css","fbe994054426fadb2dff69d824c5c67a"],["E:/GitHubBlog/public/assets/js/APlayer.min.js","8f1017e7a73737e631ff95fa51e4e7d7"],["E:/GitHubBlog/public/assets/js/Meting.min.js","bfac0368480fd344282ec018d28f173d"],["E:/GitHubBlog/public/css/404.css","b1bb50e1fabe41adcec483f6427fb3aa"],["E:/GitHubBlog/public/css/index.css","2389364df340948a5b822422956c2c87"],["E:/GitHubBlog/public/css/var.css","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/img/algolia.svg","fd40b88ac5370a5353a50b8175c1f367"],["E:/GitHubBlog/public/img/avatar.png","6cc4a809d23e3d8946a299ae4ce4e4cd"],["E:/GitHubBlog/public/index.html","ed9b3c2955df89deb8bb2bbd5741d0d3"],["E:/GitHubBlog/public/js/copy.js","45aae54bf2e43ac27ecc41eb5e0bacf7"],["E:/GitHubBlog/public/js/fancybox.js","f84d626654b9bbc05720952b3effe062"],["E:/GitHubBlog/public/js/fireworks.js","35933ce61c158ef9c33b5e089fe757ac"],["E:/GitHubBlog/public/js/head.js","347edd99f8e3921b45fa617e839d8182"],["E:/GitHubBlog/public/js/hexo-theme-melody.js","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/js/katex.js","d971ba8b8dee1120ef66744b89cfd2b1"],["E:/GitHubBlog/public/js/scroll.js","a12ad7e37b9e325f8da3d0523d3757c7"],["E:/GitHubBlog/public/js/search/algolia.js","53160985d32d6fd66d98aa0e05b081ac"],["E:/GitHubBlog/public/js/search/local-search.js","1565b508bd866ed6fbd724a3858af5d8"],["E:/GitHubBlog/public/js/sidebar.js","d24db780974e661198eeb2e45f20a28f"],["E:/GitHubBlog/public/js/third-party/anime.min.js","9b4bbe6deb700e1c3606eab732f5eea5"],["E:/GitHubBlog/public/js/third-party/canvas-ribbon.js","09c181544ddff1db701db02ac30453e0"],["E:/GitHubBlog/public/js/third-party/jquery.fancybox.min.js","3c9fa1c1199cd4f874d855ecb1641335"],["E:/GitHubBlog/public/js/third-party/jquery.min.js","c9f5aeeca3ad37bf2aa006139b935f0a"],["E:/GitHubBlog/public/js/third-party/reveal/head.min.js","aad121203010122e05f1766d92385214"],["E:/GitHubBlog/public/js/third-party/reveal/highlight.min.js","44594243bec43813a16371af8fe7e105"],["E:/GitHubBlog/public/js/third-party/reveal/markdown.min.js","7ec4cef5a7fe3f0bf0eb4dc6d7bca114"],["E:/GitHubBlog/public/js/third-party/reveal/marked.min.js","c2a88705e206d71dc21fdc4445349127"],["E:/GitHubBlog/public/js/third-party/reveal/math.min.js","0a278fee2e57c530ab07f7d2d9ea8d96"],["E:/GitHubBlog/public/js/third-party/reveal/notes.min.js","89a0dfae4d706f9c75b317f686c3aa14"],["E:/GitHubBlog/public/js/third-party/reveal/reveal.min.js","8988419d67efb5fe93e291a357e26ec9"],["E:/GitHubBlog/public/js/third-party/reveal/zoom.min.js","9791f96e63e7d534cba2b67d4bda0419"],["E:/GitHubBlog/public/js/third-party/velocity.min.js","64da069aba987ea0512cf610600a56d1"],["E:/GitHubBlog/public/js/third-party/velocity.ui.min.js","c8ca438424a080620f7b2f4ee4b0fff1"],["E:/GitHubBlog/public/js/transition.js","911db4268f0f6621073afcced9e1bfef"],["E:/GitHubBlog/public/js/utils.js","3ff3423d966a1c351e9867813b3f6d36"],["E:/GitHubBlog/public/page/2/index.html","bdb78091429b0e1a68557b8dad522977"],["E:/GitHubBlog/public/page/3/index.html","b89927e6be94563095758ff3bef9313d"],["E:/GitHubBlog/public/page/4/index.html","ef373e271179bad78eae7d4d78af2b60"],["E:/GitHubBlog/public/page/5/index.html","e3a73c5edf3b58728ed74bba37cf76d6"],["E:/GitHubBlog/public/tags/Android/index.html","2a2060ae5fa5a9e41f394714401fbfea"],["E:/GitHubBlog/public/tags/NLP/index.html","9456ca2484e1a2ddcc43ee4d065a1e84"],["E:/GitHubBlog/public/tags/NLP/page/2/index.html","95c5d4b13581d18ddb6c323a5a75c420"],["E:/GitHubBlog/public/tags/R/index.html","b214af8db107e2a610ff10d0d31b37c1"],["E:/GitHubBlog/public/tags/index.html","dfb1e168f0d210843443fac95f690a0d"],["E:/GitHubBlog/public/tags/java/index.html","7b631be5bcaef6af15a11b876a7f96f1"],["E:/GitHubBlog/public/tags/leetcode/index.html","5b70fa327db95fe10e8163bc20ee37e6"],["E:/GitHubBlog/public/tags/python/index.html","c6ce16e8affc1eebedd899bbfb774f54"],["E:/GitHubBlog/public/tags/优化方法/index.html","9bd49b6afc071df7a7e4d8f2d023f157"],["E:/GitHubBlog/public/tags/总结/index.html","1866a27acb885e2e4a42dc477384655e"],["E:/GitHubBlog/public/tags/总结/page/2/index.html","a7255582bcdcd5d7d6d90ef8f9fd18b2"],["E:/GitHubBlog/public/tags/数学/index.html","1759a38a45946c462840121d0681e2d4"],["E:/GitHubBlog/public/tags/数据分析/index.html","3e1d955e40d86986b774831578650491"],["E:/GitHubBlog/public/tags/数据挖掘/index.html","c622066d9246906e3e85afc31cd61c59"],["E:/GitHubBlog/public/tags/数据结构/index.html","d6284eda51f34a1ef62d6b388e2f8636"],["E:/GitHubBlog/public/tags/机器学习/index.html","377bfca9d47b9a660ccc6ce90aef16fd"],["E:/GitHubBlog/public/tags/深度学习/index.html","f404e1a306f540821147074ab3c77e19"],["E:/GitHubBlog/public/tags/爬虫/index.html","5a77f718b4cbe7d5354405e02d9e1921"],["E:/GitHubBlog/public/tags/笔记/index.html","28b47a34b4fcdf9b19668368f4bd9e21"],["E:/GitHubBlog/public/tags/论文/index.html","f176a7f98b34ddcf1894526fb9212962"],["E:/GitHubBlog/public/tags/论文/page/2/index.html","5033b369dd32f6b1dbfa68fa70a2bf38"],["E:/GitHubBlog/public/tags/论文/page/3/index.html","88690e877d1f8920f557760f801d7998"],["E:/GitHubBlog/public/tags/读书笔记/index.html","5ae504b93edf87e01439fea28d537805"]];
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







