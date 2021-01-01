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

var precacheConfig = [["E:/GitHubBlog/public/2020/01/28/hello-world/index.html","fa8dacbb2a1a696d30c12b113e9a6b7c"],["E:/GitHubBlog/public/2020/02/08/python爬虫学习总结/index.html","5626c6e4981b55f56035d71010979d98"],["E:/GitHubBlog/public/2020/02/19/数据结构第一周-绪论/index.html","578a3dcd79f7a0ab9f16f2b8c3d5820d"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-1/index.html","0823eb4495ab17209109606e88241f04"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-2-创建数据集/index.html","2f2e19a912773a58f34c8811c5c4d723"],["E:/GitHubBlog/public/2020/03/01/《R语言实战》学习总结-3-图形初阶/index.html","b4273d0daa9af60c1842abcfcd919b39"],["E:/GitHubBlog/public/2020/03/02/数据结构第二周-线性结构1/index.html","83d47b44f561baafde5da694aedb6818"],["E:/GitHubBlog/public/2020/03/03/《R语言实战》学习总结-4-基本数据管理/index.html","c36f45edc570d8f15fa157c10b9f45cb"],["E:/GitHubBlog/public/2020/03/14/数据结构第三周-线性结构2/index.html","dbee08f407d87a46fb4bdc64b6ef67e2"],["E:/GitHubBlog/public/2020/03/14/数据结构第四周-栈与队列/index.html","fcea9c140b3678824b235d82c0187301"],["E:/GitHubBlog/public/2020/03/17/leetcode记录-1、88、20/index.html","4a61d4daf77d4221fc50460f7f44cb63"],["E:/GitHubBlog/public/2020/03/20/温度转换程序/index.html","4cacd045fbdfb1e107042f287dbcf5e7"],["E:/GitHubBlog/public/2020/03/22/数据结构第五周-递归/index.html","072a1cf17ce764703674ef27bcd44615"],["E:/GitHubBlog/public/2020/03/22/数据结构第六周-串与数组/index.html","b01166f1afc65355d9741f4bea745a50"],["E:/GitHubBlog/public/2020/04/02/leetcode记录-52/index.html","67b48ee44e7b79b0b33abc30a74b9fb2"],["E:/GitHubBlog/public/2020/04/17/Android第八周作业-每天第一次打开应用更新数据/index.html","8d5b07f9cc6e5c4b9d3b39ccef3ad329"],["E:/GitHubBlog/public/2020/05/21/attention模型/index.html","bb5fa8b310d1b482acb7a91791c07969"],["E:/GitHubBlog/public/2020/05/21/编码器与解码器框架/index.html","4e3dbcea54bc703e5a649543d77edd18"],["E:/GitHubBlog/public/2020/07/11/思考，快与慢/index.html","3e71f5fab246b918e7d5f9696b9d2153"],["E:/GitHubBlog/public/2020/07/13/20200713-0717总结/index.html","ed6df51a8186063be2fe9c6732cb8200"],["E:/GitHubBlog/public/2020/07/13/ZeroShotCeres Zero-Shot Relation Extraction/index.html","de9be49f2ddb50a5285330a1fde44765"],["E:/GitHubBlog/public/2020/07/20/Keyphrase Generation A Text Summarization Struggle/index.html","ba713fb699f0caeaaec10c02f01e6840"],["E:/GitHubBlog/public/2020/07/20/TPR/index.html","74341fd25c532272e1fdae6bce0b6a73"],["E:/GitHubBlog/public/2020/07/23/Keyphrase Generation for Scientific Document Retrieval/index.html","c8114fe2cfa334e410ca16ff27a340fc"],["E:/GitHubBlog/public/2020/07/24/20200718-0724总结/index.html","a3e98558cd1145603cb58659c52a1a1e"],["E:/GitHubBlog/public/2020/07/31/20200725-0731总结/index.html","d4c4201015d8989f2b611136242b761c"],["E:/GitHubBlog/public/2020/08/07/20200801-0807总结/index.html","8ab7dc055617c2e69a022a4aa84d8e60"],["E:/GitHubBlog/public/2020/08/12/逻辑回归算法/index.html","2412601dac54f697077305aa9d5a6d0a"],["E:/GitHubBlog/public/2020/08/13/李宏毅-机器学习21/index.html","9017d8ee71d12421afd8b571daa0a086"],["E:/GitHubBlog/public/2020/08/14/20200808-0815总结/index.html","e8293863e0d53915de226b644948cb3a"],["E:/GitHubBlog/public/2020/09/03/20200816-0903总结/index.html","45f69ff5044b2aa9b2fa51b3436eb016"],["E:/GitHubBlog/public/2020/09/16/20200914-0918总结/index.html","c04b818169628f0e230a14bb41c22399"],["E:/GitHubBlog/public/2020/09/25/20200919-0925总结/index.html","76738fafb0b5efeed4a21608db1d6cd5"],["E:/GitHubBlog/public/2020/10/06/《Deep Keyphrase Generation》详细解析！/index.html","378debcc331eabb57d2b3a99629cb41d"],["E:/GitHubBlog/public/2020/10/16/20201008-1017总结/index.html","5af4580ed96b26cca78f48f98f84e229"],["E:/GitHubBlog/public/2020/10/20/特征选择的几种方法(一)————Filter过滤法/index.html","6051ebfc4132da931a42612d3dcd66a7"],["E:/GitHubBlog/public/2020/10/21/PCA/index.html","9039974d80047bff7a2da6d055357f50"],["E:/GitHubBlog/public/2020/10/22/20201018-1023总结/index.html","8121cd3b472999b65c8059a58ea5e2a4"],["E:/GitHubBlog/public/2020/10/24/20201024-1030/index.html","9eda66db694b1b7e859a07521280340b"],["E:/GitHubBlog/public/2020/10/30/1030讨论记录/index.html","c59414186f2df8de81ae1443a71ce3d1"],["E:/GitHubBlog/public/2020/11/03/20201103-总结/index.html","e814646c66fc3e7e5e077d46fa4cab3d"],["E:/GitHubBlog/public/2020/11/05/线性回归与逻辑回归/index.html","5947c9fb10edb7e7b6e49fb5c6e537ab"],["E:/GitHubBlog/public/2020/11/07/决策树/index.html","2bab812b3e89632e083143eb5f427912"],["E:/GitHubBlog/public/2020/11/13/20201107-1113总结/index.html","d1059c6cea8d098dfdb4725ce2256d57"],["E:/GitHubBlog/public/2020/11/15/20201114-119总结/index.html","b54757226a16d3cbb6892c1c1f2fcd93"],["E:/GitHubBlog/public/2020/11/16/手推一个SVM/index.html","34451b45a5f95fe8fc92c057e961b343"],["E:/GitHubBlog/public/2020/11/27/20201127左右的总结/index.html","830baf599b797a5c68a5c8020aee36b1"],["E:/GitHubBlog/public/2020/11/30/GSA的数值实验/index.html","806d8b6ba836a21406f81f3ab0bdb814"],["E:/GitHubBlog/public/2020/12/23/第一个深度学习模型（pytorch）/index.html","04345c1df70655572539ccda7af08d66"],["E:/GitHubBlog/public/2021/01/02/鲍鱼最终版/index.html","7779c7b8c94963b4cb46b3f49a2815a2"],["E:/GitHubBlog/public/archives/2020/01/index.html","0a0fd0f53527b3cfadbaa36ebab7492f"],["E:/GitHubBlog/public/archives/2020/02/index.html","f3a2f943bc43b332d383acde9f763fa7"],["E:/GitHubBlog/public/archives/2020/03/index.html","96bfdaf45961bc789750210f54b78d01"],["E:/GitHubBlog/public/archives/2020/04/index.html","af5e41cd77ba3ca503d8f3f02ab97e0b"],["E:/GitHubBlog/public/archives/2020/05/index.html","046071c081063b3a79a7871aa5ff5d8e"],["E:/GitHubBlog/public/archives/2020/07/index.html","b4aca2711ce73e35e8f2352a5ca5c9d1"],["E:/GitHubBlog/public/archives/2020/08/index.html","9a4d700f1fc9524c867e9d0dff0daa5e"],["E:/GitHubBlog/public/archives/2020/09/index.html","9c2abb58b6d1103dc7fb9f5076d3ee95"],["E:/GitHubBlog/public/archives/2020/10/index.html","f95e52697769f8a241c258a28b5ddcb8"],["E:/GitHubBlog/public/archives/2020/11/index.html","608346b3f5dcfa9b06e2e3b794fdcaef"],["E:/GitHubBlog/public/archives/2020/12/index.html","230adc94be4f23577008f0e173221e45"],["E:/GitHubBlog/public/archives/2020/index.html","765feff390698e1ebc4c49521d2bc181"],["E:/GitHubBlog/public/archives/2020/page/2/index.html","cd05eade1063e2473d6d3a445dc90844"],["E:/GitHubBlog/public/archives/2020/page/3/index.html","0880a9282b854646d57d32dcf3823251"],["E:/GitHubBlog/public/archives/2020/page/4/index.html","b7147b41672c3dbc8b59aeaf3fde6996"],["E:/GitHubBlog/public/archives/2020/page/5/index.html","4d256ecbb8bd520fa7808bb325e3736a"],["E:/GitHubBlog/public/archives/2021/01/index.html","80ff8be362d121f0f5cf387c1873f417"],["E:/GitHubBlog/public/archives/2021/index.html","ce619b92ac554c127303b5a312e48b0a"],["E:/GitHubBlog/public/archives/index.html","ecd50d6a071303c5be9141740bd6e47a"],["E:/GitHubBlog/public/archives/page/2/index.html","61ea638e52832882f2fb1b7c22648d69"],["E:/GitHubBlog/public/archives/page/3/index.html","52fa8bb2a754668888961703c2791473"],["E:/GitHubBlog/public/archives/page/4/index.html","e62ae4f4cf0caebd4db77ada4f5ff047"],["E:/GitHubBlog/public/archives/page/5/index.html","c85e39c72ca41b65859bd8f720b75dc1"],["E:/GitHubBlog/public/assets/css/APlayer.min.css","fbe994054426fadb2dff69d824c5c67a"],["E:/GitHubBlog/public/assets/js/APlayer.min.js","8f1017e7a73737e631ff95fa51e4e7d7"],["E:/GitHubBlog/public/assets/js/Meting.min.js","bfac0368480fd344282ec018d28f173d"],["E:/GitHubBlog/public/css/404.css","b1bb50e1fabe41adcec483f6427fb3aa"],["E:/GitHubBlog/public/css/index.css","2389364df340948a5b822422956c2c87"],["E:/GitHubBlog/public/css/var.css","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/img/algolia.svg","fd40b88ac5370a5353a50b8175c1f367"],["E:/GitHubBlog/public/img/avatar.png","6cc4a809d23e3d8946a299ae4ce4e4cd"],["E:/GitHubBlog/public/index.html","5bac95eafaec9c4fb83e45bc74670076"],["E:/GitHubBlog/public/js/copy.js","45aae54bf2e43ac27ecc41eb5e0bacf7"],["E:/GitHubBlog/public/js/fancybox.js","f84d626654b9bbc05720952b3effe062"],["E:/GitHubBlog/public/js/fireworks.js","35933ce61c158ef9c33b5e089fe757ac"],["E:/GitHubBlog/public/js/head.js","347edd99f8e3921b45fa617e839d8182"],["E:/GitHubBlog/public/js/hexo-theme-melody.js","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/js/katex.js","d971ba8b8dee1120ef66744b89cfd2b1"],["E:/GitHubBlog/public/js/scroll.js","a12ad7e37b9e325f8da3d0523d3757c7"],["E:/GitHubBlog/public/js/search/algolia.js","53160985d32d6fd66d98aa0e05b081ac"],["E:/GitHubBlog/public/js/search/local-search.js","1565b508bd866ed6fbd724a3858af5d8"],["E:/GitHubBlog/public/js/sidebar.js","d24db780974e661198eeb2e45f20a28f"],["E:/GitHubBlog/public/js/third-party/anime.min.js","9b4bbe6deb700e1c3606eab732f5eea5"],["E:/GitHubBlog/public/js/third-party/canvas-ribbon.js","09c181544ddff1db701db02ac30453e0"],["E:/GitHubBlog/public/js/third-party/jquery.fancybox.min.js","3c9fa1c1199cd4f874d855ecb1641335"],["E:/GitHubBlog/public/js/third-party/jquery.min.js","c9f5aeeca3ad37bf2aa006139b935f0a"],["E:/GitHubBlog/public/js/third-party/reveal/head.min.js","aad121203010122e05f1766d92385214"],["E:/GitHubBlog/public/js/third-party/reveal/highlight.min.js","44594243bec43813a16371af8fe7e105"],["E:/GitHubBlog/public/js/third-party/reveal/markdown.min.js","7ec4cef5a7fe3f0bf0eb4dc6d7bca114"],["E:/GitHubBlog/public/js/third-party/reveal/marked.min.js","c2a88705e206d71dc21fdc4445349127"],["E:/GitHubBlog/public/js/third-party/reveal/math.min.js","0a278fee2e57c530ab07f7d2d9ea8d96"],["E:/GitHubBlog/public/js/third-party/reveal/notes.min.js","89a0dfae4d706f9c75b317f686c3aa14"],["E:/GitHubBlog/public/js/third-party/reveal/reveal.min.js","8988419d67efb5fe93e291a357e26ec9"],["E:/GitHubBlog/public/js/third-party/reveal/zoom.min.js","9791f96e63e7d534cba2b67d4bda0419"],["E:/GitHubBlog/public/js/third-party/velocity.min.js","64da069aba987ea0512cf610600a56d1"],["E:/GitHubBlog/public/js/third-party/velocity.ui.min.js","c8ca438424a080620f7b2f4ee4b0fff1"],["E:/GitHubBlog/public/js/transition.js","911db4268f0f6621073afcced9e1bfef"],["E:/GitHubBlog/public/js/utils.js","3ff3423d966a1c351e9867813b3f6d36"],["E:/GitHubBlog/public/page/2/index.html","760ba0c5081fe9ec57c6b476b204f9c4"],["E:/GitHubBlog/public/page/3/index.html","c2f4947ebb7c0cd5f86f2729b5caad4f"],["E:/GitHubBlog/public/page/4/index.html","47a3ae245672dacf11ae5a6ecfa4e34a"],["E:/GitHubBlog/public/page/5/index.html","bf0b6338efbfb55a97ce084f071bfd33"],["E:/GitHubBlog/public/tags/Android/index.html","0e475c87c21ea851ed6ee3113b6b9bab"],["E:/GitHubBlog/public/tags/NLP/index.html","d77c9463cf95ec72bf8fc8199d35c8a9"],["E:/GitHubBlog/public/tags/NLP/page/2/index.html","5936f44b79cc31df141ef6cc3f528c27"],["E:/GitHubBlog/public/tags/NLP/page/3/index.html","626584d4b22395583119e3733494c15e"],["E:/GitHubBlog/public/tags/R/index.html","9dd8567af1aee49f3adb96a13412b4c6"],["E:/GitHubBlog/public/tags/index.html","b140d4fffad23d2172cf09e170943520"],["E:/GitHubBlog/public/tags/java/index.html","650a1c0e8442df1afdb14bfd2e00ee9d"],["E:/GitHubBlog/public/tags/leetcode/index.html","80fd62ea9250098fd66b5a825c952dbf"],["E:/GitHubBlog/public/tags/python/index.html","a79ecbffc30d415383ade09c27862b76"],["E:/GitHubBlog/public/tags/pytorch/index.html","79ecb18cc0a9f758a3ed3e61b9b15919"],["E:/GitHubBlog/public/tags/总结/index.html","573069fd081ce0f00bd9b6b8ed5a440e"],["E:/GitHubBlog/public/tags/总结/page/2/index.html","5fe435c4c89b638c24f793e644078356"],["E:/GitHubBlog/public/tags/数据分析/index.html","15469939b0431d21d42d1ae3537594ce"],["E:/GitHubBlog/public/tags/数据挖掘/index.html","2850ac54d975fe8209345252f06e63ea"],["E:/GitHubBlog/public/tags/数据结构/index.html","c17bf3391e5244af2856159e8efc316e"],["E:/GitHubBlog/public/tags/机器学习/index.html","be2a9d646d17c8630738bd3ffc3bf0ce"],["E:/GitHubBlog/public/tags/深度学习/index.html","d681d201cc69f17f3bb57c3c11aa5430"],["E:/GitHubBlog/public/tags/爬虫/index.html","c5ae964fa75b135fe155cd6393151e0a"],["E:/GitHubBlog/public/tags/笔记/index.html","914450c3f6c868376cc5ec12c0b60f32"],["E:/GitHubBlog/public/tags/论文/index.html","910738a25abed6e5471350ebeaf19e80"],["E:/GitHubBlog/public/tags/论文/page/2/index.html","1a80b1c9bd1dc724bb3fc86eb0e65dff"],["E:/GitHubBlog/public/tags/论文/page/3/index.html","bb1618bfbb93b0e0180b7dc75498fe91"],["E:/GitHubBlog/public/tags/读书笔记/index.html","6a8cdf781c4f8877724505b086fe6e52"]];
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







