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

var precacheConfig = [["E:/GitHubBlog/public/2020/01/28/hello-world/index.html","1c8c0f780a75b44241f6878861de8170"],["E:/GitHubBlog/public/2020/02/08/python爬虫学习总结/index.html","d684c86d15fce34de36d73f997b73ca8"],["E:/GitHubBlog/public/2020/02/19/数据结构第一周-绪论/index.html","2ac89402ff5df82924b9c9464531d2ee"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-1/index.html","4bb593c13bbc0a8d608247972f218a77"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-2-创建数据集/index.html","c77c73211cbb750966b98eabb9c45330"],["E:/GitHubBlog/public/2020/03/01/《R语言实战》学习总结-3-图形初阶/index.html","03b9f211eace148aee6eeb89377370ff"],["E:/GitHubBlog/public/2020/03/02/数据结构第二周-线性结构1/index.html","54271827e3bd650d1116a10ebc83c8d5"],["E:/GitHubBlog/public/2020/03/03/《R语言实战》学习总结-4-基本数据管理/index.html","9dfdb7390c0f9b2f6a6a3975d5bd4e44"],["E:/GitHubBlog/public/2020/03/14/数据结构第三周-线性结构2/index.html","2d3c6a2cd50785becde274497e177cf3"],["E:/GitHubBlog/public/2020/03/14/数据结构第四周-栈与队列/index.html","7b47926d0f2454feb3560ab3181f7aaa"],["E:/GitHubBlog/public/2020/03/17/leetcode记录-1、88、20/index.html","9d5b6a12ea702de21e5272d09961fcf2"],["E:/GitHubBlog/public/2020/03/20/温度转换程序/index.html","59db1984970d85aede1e75d5622df608"],["E:/GitHubBlog/public/2020/03/22/数据结构第五周-递归/index.html","eab8846cec376c15bb5918e17b0088ff"],["E:/GitHubBlog/public/2020/03/22/数据结构第六周-串与数组/index.html","c1db9f3d95868f80df7317a633057c3e"],["E:/GitHubBlog/public/2020/04/02/leetcode记录-52/index.html","c87bdeb2daf695467d42e829c8f40bca"],["E:/GitHubBlog/public/2020/04/17/Android第八周作业-每天第一次打开应用更新数据/index.html","6c652fa467e7d1f9f84adf5b244df6e1"],["E:/GitHubBlog/public/2020/05/21/attention模型/index.html","b2eb7c674d71e923c2248dbf0c1730f6"],["E:/GitHubBlog/public/2020/05/21/编码器与解码器框架/index.html","0520337c118ee5eb67008cc85782de14"],["E:/GitHubBlog/public/2020/07/11/思考，快与慢/index.html","e0a459a2ad4350646a4ce08a1841bd67"],["E:/GitHubBlog/public/2020/07/13/20200713-0717总结/index.html","1d579a2957c6c55b8af3fdaeb18ef63c"],["E:/GitHubBlog/public/2020/07/13/ZeroShotCeres Zero-Shot Relation Extraction/index.html","c84f55b977a450d3e64982e54a620c63"],["E:/GitHubBlog/public/2020/07/20/Keyphrase Generation A Text Summarization Struggle/index.html","4158132aa11f97b747e8f838dd9a7e96"],["E:/GitHubBlog/public/2020/07/20/TPR/index.html","418bf1a178cbde1fbfbdc888d9591ff8"],["E:/GitHubBlog/public/2020/07/23/Keyphrase Generation for Scientific Document Retrieval/index.html","4bdd5c9cd1c209e4f5094c1e94d724bd"],["E:/GitHubBlog/public/2020/07/24/20200718-0724总结/index.html","0f3c9173bba3b453d8ed522dbc0839a3"],["E:/GitHubBlog/public/2020/07/31/20200725-0731总结/index.html","fd4bfcdff6de889b2c7a2f2c12b994f4"],["E:/GitHubBlog/public/2020/08/07/20200801-0807总结/index.html","4655cd603d574d43970b432c3fdf9841"],["E:/GitHubBlog/public/2020/08/12/逻辑回归算法/index.html","55d532cf6a2f8796437519ce7d3add94"],["E:/GitHubBlog/public/2020/08/13/李宏毅-机器学习21/index.html","768d234293c93c9cffe1d684b4fbabe7"],["E:/GitHubBlog/public/2020/08/14/20200808-0815总结/index.html","a66c114f51ec8293c9c66983709a84a2"],["E:/GitHubBlog/public/2020/09/03/20200816-0903总结/index.html","bf46aa0839845f48a6c395b665ae3bef"],["E:/GitHubBlog/public/2020/09/16/20200914-0918总结/index.html","5df46843375ca577cbcdcd8481a78484"],["E:/GitHubBlog/public/2020/09/25/20200919-0925总结/index.html","93c9b6ae33ea840e83260f1b57fad175"],["E:/GitHubBlog/public/2020/10/06/《Deep Keyphrase Generation》详细解析！/index.html","4e3132db2e520a2337b2fe8eba23c4f6"],["E:/GitHubBlog/public/2020/10/16/20201008-1017总结/index.html","88c181ab9c89f05eee6e4633befee742"],["E:/GitHubBlog/public/2020/10/20/特征选择的几种方法(一)————Filter过滤法/index.html","597daa6328b729571ab68a1a57ddbe10"],["E:/GitHubBlog/public/2020/10/21/PCA/index.html","fbd80ae7483211564fcb8f614385999b"],["E:/GitHubBlog/public/2020/10/22/20201018-1023总结/index.html","ad68370057bad252d451de7d6faa5559"],["E:/GitHubBlog/public/2020/10/24/20201024-1030/index.html","bc9a995635fac1976886283b7d975b63"],["E:/GitHubBlog/public/2020/10/30/1030讨论记录/index.html","bec80471fcb791fcf479c778dae43ccd"],["E:/GitHubBlog/public/2020/11/03/20201103-总结/index.html","21979d915d8f05e388af1f0b32bc9593"],["E:/GitHubBlog/public/2020/11/05/线性回归与逻辑回归/index.html","41d721b5a104aa4f1e22e57fe1195e31"],["E:/GitHubBlog/public/2020/11/07/决策树/index.html","ce7323a2daba9208fb930ff6a928433e"],["E:/GitHubBlog/public/2020/11/13/20201107-1113总结/index.html","23ba154d631849d9251224661230e5cc"],["E:/GitHubBlog/public/2020/11/15/20201114-119总结/index.html","890c40e09681bfa50e6279a80da0b383"],["E:/GitHubBlog/public/2020/11/16/手推一个SVM/index.html","f009313abd12023dfb9c3e5b3136235e"],["E:/GitHubBlog/public/2020/11/27/20201127左右的总结/index.html","2b0122b04816820de4394e4a6d9f066f"],["E:/GitHubBlog/public/archives/2020/01/index.html","42a2455efba5c6bdc24f6075568050e7"],["E:/GitHubBlog/public/archives/2020/02/index.html","9cef1ecba54ed26114dd4f5c9643400d"],["E:/GitHubBlog/public/archives/2020/03/index.html","18805cf049dcf9b25c0f1d353ab75d79"],["E:/GitHubBlog/public/archives/2020/04/index.html","db40b3bebee2ae671f16d630ecfe4cac"],["E:/GitHubBlog/public/archives/2020/05/index.html","b30cc29e20fa7fb23c6684342b80d625"],["E:/GitHubBlog/public/archives/2020/07/index.html","542608413844b0585c54087b4866997f"],["E:/GitHubBlog/public/archives/2020/08/index.html","7e812ca8909d0b9db89a4765801fc592"],["E:/GitHubBlog/public/archives/2020/09/index.html","92a988fa4ec52f0a5d7b93705865a22c"],["E:/GitHubBlog/public/archives/2020/10/index.html","86b857584e21ce4090a64f664f0f7012"],["E:/GitHubBlog/public/archives/2020/11/index.html","0dfa2d5a096ac5807e9326f5f2563d1a"],["E:/GitHubBlog/public/archives/2020/index.html","eaecc1aaac29970f5b600af4dd898cf0"],["E:/GitHubBlog/public/archives/2020/page/2/index.html","9105c53c25bc8006dbca6d5d52d6ecba"],["E:/GitHubBlog/public/archives/2020/page/3/index.html","9afc131a8d9423552bc5cdebaac8436d"],["E:/GitHubBlog/public/archives/2020/page/4/index.html","fc6cb9f273e4f1661325a61c68027f81"],["E:/GitHubBlog/public/archives/2020/page/5/index.html","08803c587720f9f077ae3715f59b66de"],["E:/GitHubBlog/public/archives/index.html","20111dba34cfa7db7028faa0a181d46b"],["E:/GitHubBlog/public/archives/page/2/index.html","542d3eef5a6bfd17520303fcb6d36a38"],["E:/GitHubBlog/public/archives/page/3/index.html","e77c8bba8a24cf23a98d1f16876f7245"],["E:/GitHubBlog/public/archives/page/4/index.html","4d55eaaaf091b3c525f95acafc228746"],["E:/GitHubBlog/public/archives/page/5/index.html","7537db5b4a824c578546060e01951261"],["E:/GitHubBlog/public/assets/css/APlayer.min.css","fbe994054426fadb2dff69d824c5c67a"],["E:/GitHubBlog/public/assets/js/APlayer.min.js","8f1017e7a73737e631ff95fa51e4e7d7"],["E:/GitHubBlog/public/assets/js/Meting.min.js","bfac0368480fd344282ec018d28f173d"],["E:/GitHubBlog/public/css/404.css","b1bb50e1fabe41adcec483f6427fb3aa"],["E:/GitHubBlog/public/css/index.css","2389364df340948a5b822422956c2c87"],["E:/GitHubBlog/public/css/var.css","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/img/algolia.svg","fd40b88ac5370a5353a50b8175c1f367"],["E:/GitHubBlog/public/img/avatar.png","6cc4a809d23e3d8946a299ae4ce4e4cd"],["E:/GitHubBlog/public/index.html","58fce1e72fe7ec10d7f6659ce723d673"],["E:/GitHubBlog/public/js/copy.js","45aae54bf2e43ac27ecc41eb5e0bacf7"],["E:/GitHubBlog/public/js/fancybox.js","f84d626654b9bbc05720952b3effe062"],["E:/GitHubBlog/public/js/fireworks.js","35933ce61c158ef9c33b5e089fe757ac"],["E:/GitHubBlog/public/js/head.js","347edd99f8e3921b45fa617e839d8182"],["E:/GitHubBlog/public/js/hexo-theme-melody.js","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/js/katex.js","d971ba8b8dee1120ef66744b89cfd2b1"],["E:/GitHubBlog/public/js/scroll.js","a12ad7e37b9e325f8da3d0523d3757c7"],["E:/GitHubBlog/public/js/search/algolia.js","53160985d32d6fd66d98aa0e05b081ac"],["E:/GitHubBlog/public/js/search/local-search.js","1565b508bd866ed6fbd724a3858af5d8"],["E:/GitHubBlog/public/js/sidebar.js","d24db780974e661198eeb2e45f20a28f"],["E:/GitHubBlog/public/js/third-party/anime.min.js","9b4bbe6deb700e1c3606eab732f5eea5"],["E:/GitHubBlog/public/js/third-party/canvas-ribbon.js","09c181544ddff1db701db02ac30453e0"],["E:/GitHubBlog/public/js/third-party/jquery.fancybox.min.js","3c9fa1c1199cd4f874d855ecb1641335"],["E:/GitHubBlog/public/js/third-party/jquery.min.js","c9f5aeeca3ad37bf2aa006139b935f0a"],["E:/GitHubBlog/public/js/third-party/reveal/head.min.js","aad121203010122e05f1766d92385214"],["E:/GitHubBlog/public/js/third-party/reveal/highlight.min.js","44594243bec43813a16371af8fe7e105"],["E:/GitHubBlog/public/js/third-party/reveal/markdown.min.js","7ec4cef5a7fe3f0bf0eb4dc6d7bca114"],["E:/GitHubBlog/public/js/third-party/reveal/marked.min.js","c2a88705e206d71dc21fdc4445349127"],["E:/GitHubBlog/public/js/third-party/reveal/math.min.js","0a278fee2e57c530ab07f7d2d9ea8d96"],["E:/GitHubBlog/public/js/third-party/reveal/notes.min.js","89a0dfae4d706f9c75b317f686c3aa14"],["E:/GitHubBlog/public/js/third-party/reveal/reveal.min.js","8988419d67efb5fe93e291a357e26ec9"],["E:/GitHubBlog/public/js/third-party/reveal/zoom.min.js","9791f96e63e7d534cba2b67d4bda0419"],["E:/GitHubBlog/public/js/third-party/velocity.min.js","64da069aba987ea0512cf610600a56d1"],["E:/GitHubBlog/public/js/third-party/velocity.ui.min.js","c8ca438424a080620f7b2f4ee4b0fff1"],["E:/GitHubBlog/public/js/transition.js","911db4268f0f6621073afcced9e1bfef"],["E:/GitHubBlog/public/js/utils.js","3ff3423d966a1c351e9867813b3f6d36"],["E:/GitHubBlog/public/page/2/index.html","d06be206d90609167782b3bd980bf6ad"],["E:/GitHubBlog/public/page/3/index.html","9c7ca440806f18b804a268e7f548da0c"],["E:/GitHubBlog/public/page/4/index.html","4984641da4b58178a7952b4ed36e9c9c"],["E:/GitHubBlog/public/page/5/index.html","97d4870b0e259a8a565c48bfe287e464"],["E:/GitHubBlog/public/tags/Android/index.html","688f845e62d056318fcff6c3f4ea00ec"],["E:/GitHubBlog/public/tags/NLP/index.html","82cbf3451830fa3f3a1f42d25ca5a6f4"],["E:/GitHubBlog/public/tags/NLP/page/2/index.html","b0c7d9845abe67457ca3dc80b260a752"],["E:/GitHubBlog/public/tags/R/index.html","430c22ea3044dd7de545082d8a860300"],["E:/GitHubBlog/public/tags/index.html","0a14dbcb825a22193eb310f42b53ac07"],["E:/GitHubBlog/public/tags/java/index.html","04002214b56fb61ffe29f58b66318f3c"],["E:/GitHubBlog/public/tags/leetcode/index.html","1cdefccc553ab47a24669072279f3e04"],["E:/GitHubBlog/public/tags/python/index.html","31e6f5ae4229bbb4855f1410699b4812"],["E:/GitHubBlog/public/tags/总结/index.html","9affe2b0fa21a0a188b6d30ed5797015"],["E:/GitHubBlog/public/tags/总结/page/2/index.html","5fea251ca837fcd65530c991d70b050a"],["E:/GitHubBlog/public/tags/数据分析/index.html","6633ad3fc2239efe3597c136f536f174"],["E:/GitHubBlog/public/tags/数据挖掘/index.html","0cd467720f2b0a22fb313652fe513f20"],["E:/GitHubBlog/public/tags/数据结构/index.html","52f606cf59a166d631c8992c5b0535f3"],["E:/GitHubBlog/public/tags/机器学习/index.html","8156e9850b967f6bf24f429cec6564bc"],["E:/GitHubBlog/public/tags/深度学习/index.html","5d411ccff8c43ef5b64ea4e8e04c364f"],["E:/GitHubBlog/public/tags/爬虫/index.html","dd48a637e4e714a7d127c5c9d722f098"],["E:/GitHubBlog/public/tags/笔记/index.html","8d5a74e80fa1faf3041994c772a307dc"],["E:/GitHubBlog/public/tags/论文/index.html","1f6275aab951ff45bc28208a065c3393"],["E:/GitHubBlog/public/tags/论文/page/2/index.html","e231f6fde2d6616d3a5af2b77c74fc9a"],["E:/GitHubBlog/public/tags/论文/page/3/index.html","d1d14af431b37314653530d1eec01991"],["E:/GitHubBlog/public/tags/读书笔记/index.html","405524da30cc36fea9301b01aff290ae"]];
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







