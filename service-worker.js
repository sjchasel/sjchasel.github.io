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

var precacheConfig = [["E:/GitHubBlog/public/2020/01/28/hello-world/index.html","d3dd355b12c933380c336aeb3f9fa0ed"],["E:/GitHubBlog/public/2020/02/08/python爬虫学习总结/index.html","7e46fe992250731727c04efa5bfba421"],["E:/GitHubBlog/public/2020/02/19/数据结构第一周-绪论/index.html","a186e6d228d5a40372a4de405bb2190d"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-1/index.html","f4e6fd7aab85b23e87499c4c241161fd"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-2-创建数据集/index.html","254de0e6c7d71eb939853a56483b9e03"],["E:/GitHubBlog/public/2020/03/01/《R语言实战》学习总结-3-图形初阶/index.html","8ac6853f4801e667617d02f3f0426a95"],["E:/GitHubBlog/public/2020/03/02/数据结构第二周-线性结构1/index.html","7507df711ada0a660f52d396b169216a"],["E:/GitHubBlog/public/2020/03/03/《R语言实战》学习总结-4-基本数据管理/index.html","73a01ebed63a27f4151be61d8ba57947"],["E:/GitHubBlog/public/2020/03/14/数据结构第三周-线性结构2/index.html","71a27d0ffc1b604ce56ac016966100b0"],["E:/GitHubBlog/public/2020/03/14/数据结构第四周-栈与队列/index.html","ebd9faa0a4c74d65b0eb26d513244fa6"],["E:/GitHubBlog/public/2020/03/17/leetcode记录-1、88、20/index.html","87202887f1321414799531dd3ab8a3cd"],["E:/GitHubBlog/public/2020/03/20/温度转换程序/index.html","e63f1289effa8660292578d169d37fdb"],["E:/GitHubBlog/public/2020/03/22/数据结构第五周-递归/index.html","8a8b4a3bfade7d9c0babf159b8f6d32e"],["E:/GitHubBlog/public/2020/03/22/数据结构第六周-串与数组/index.html","54a1dbdd67e1bb3d18d3518af04ebfea"],["E:/GitHubBlog/public/2020/04/02/leetcode记录-52/index.html","b6cd566667614623ef89f662162e295c"],["E:/GitHubBlog/public/2020/04/17/Android第八周作业-每天第一次打开应用更新数据/index.html","23fac401f1b4f7c3cdb44c127a5f7f5e"],["E:/GitHubBlog/public/2020/05/21/attention模型/index.html","ccf4f6c82401528289196d4720048dce"],["E:/GitHubBlog/public/2020/05/21/编码器与解码器框架/index.html","e0b3f8326e2e96241318c2e99501859b"],["E:/GitHubBlog/public/2020/07/11/思考，快与慢/index.html","7ddd9341033a5e013b8ec05cf0369388"],["E:/GitHubBlog/public/2020/07/13/20200713-0717总结/index.html","e15e11b9e0f868b1381e53cff9faa095"],["E:/GitHubBlog/public/2020/07/13/ZeroShotCeres Zero-Shot Relation Extraction/index.html","93d878a13ecf49497e8fb75c72e18b3c"],["E:/GitHubBlog/public/2020/07/20/Keyphrase Generation A Text Summarization Struggle/index.html","5e8cac07f15a7586420ca140ae097331"],["E:/GitHubBlog/public/2020/07/20/TPR/index.html","4a2e4c11a73a201a18a91d5200f36e69"],["E:/GitHubBlog/public/2020/07/23/Keyphrase Generation for Scientific Document Retrieval/index.html","af77b3bec69f3565b349068bd020c040"],["E:/GitHubBlog/public/2020/07/24/20200718-0724总结/index.html","80e69088211781ae0705e4adbf36557d"],["E:/GitHubBlog/public/2020/07/31/20200725-0731总结/index.html","dbb55b9c3639b0e4f4186e35f6d64a28"],["E:/GitHubBlog/public/2020/08/07/20200801-0807总结/index.html","2e8506c4750c660c616e3e9f02201e66"],["E:/GitHubBlog/public/2020/08/12/逻辑回归算法/index.html","7aaccd2e43a211b827b927dc866ca3a9"],["E:/GitHubBlog/public/2020/08/13/李宏毅-机器学习21/index.html","cc8478f3e0039ccc9566ba07a605eea7"],["E:/GitHubBlog/public/2020/08/14/20200808-0815总结/index.html","640c7bfc9dab8e071422cd780e827410"],["E:/GitHubBlog/public/2020/09/03/20200816-0903总结/index.html","ee7259626489e2f162fb36b4ee2502ac"],["E:/GitHubBlog/public/2020/09/16/20200914-0918总结/index.html","7868d5c04693ca2d32c5ce55735b0f78"],["E:/GitHubBlog/public/2020/09/25/20200919-0925总结/index.html","07f4206595fd970beb2dfd4825b87e3c"],["E:/GitHubBlog/public/2020/10/06/《Deep Keyphrase Generation》详细解析！/index.html","8854fb8420c62584e3d924b6b405b15e"],["E:/GitHubBlog/public/2020/10/16/20201008-1017总结/index.html","a52a1b129157b02f0c91a291992b9aae"],["E:/GitHubBlog/public/2020/10/20/特征选择的几种方法(一)————Filter过滤法/index.html","3bf0eb729985a93025d68fcc2fef8e6a"],["E:/GitHubBlog/public/2020/10/21/PCA/index.html","e9aae403e965d5b9962f6d575509c896"],["E:/GitHubBlog/public/2020/10/22/20201018-1023总结/index.html","1304cbcc13b8b63b13a368e18de492c5"],["E:/GitHubBlog/public/2020/10/24/20201024-1030/index.html","06be20a79aca3bd8aa097e982c30d7c6"],["E:/GitHubBlog/public/2020/10/30/1030讨论记录/index.html","4e17c53fd068d85edda4261978a47c16"],["E:/GitHubBlog/public/2020/11/03/20201103-总结/index.html","a7ce50273ee9635743c54a2deb975e84"],["E:/GitHubBlog/public/2020/11/05/线性回归与逻辑回归/index.html","de6db64aff1c92e667efc666f0f30f5e"],["E:/GitHubBlog/public/2020/11/07/决策树/index.html","4d89092693e64102a86a5ea3a1f39a83"],["E:/GitHubBlog/public/2020/11/13/20201107-1113总结/index.html","7561b1a4663f8a608fe920b0f870439a"],["E:/GitHubBlog/public/2020/11/15/20201114-119总结/index.html","ffaea92d437e0752df7a27919acc7687"],["E:/GitHubBlog/public/2020/11/16/手推一个SVM/index.html","20921eebfd53b44aac917ddd53599b9b"],["E:/GitHubBlog/public/2020/11/27/20201127左右的总结/index.html","d80a8cfd8fc0a6f78cf48da9d20cb843"],["E:/GitHubBlog/public/2020/12/23/第一个深度学习模型（pytorch）/index.html","18aabe0c04390b2404f895da3c8564fe"],["E:/GitHubBlog/public/2021/01/02/鲍鱼最终版/index.html","d36862fd4e3d6d948e1d24f157f6f904"],["E:/GitHubBlog/public/2021/01/08/GSA_NumericalExperiment/index.html","a94a850b243f742308bbe91228556ad1"],["E:/GitHubBlog/public/2021/01/16/20210115-01总结/index.html","feac4c5c6f4b1163d1e49f76a193c7e9"],["E:/GitHubBlog/public/archives/2020/01/index.html","5c78dcc3fc782541baebdb1baedb2693"],["E:/GitHubBlog/public/archives/2020/02/index.html","4cc5054faa66a8c8e364769830fb2078"],["E:/GitHubBlog/public/archives/2020/03/index.html","8dd88ca945e524282ecaa88b1475fa5a"],["E:/GitHubBlog/public/archives/2020/04/index.html","a32dc8813510393fae61b9d489ff9dfa"],["E:/GitHubBlog/public/archives/2020/05/index.html","e30e4c8632de4f44f4855529c5842872"],["E:/GitHubBlog/public/archives/2020/07/index.html","4eac971a7445d36594e74261eb680495"],["E:/GitHubBlog/public/archives/2020/08/index.html","51f17ac0ba856eff35dfc9103d29b29a"],["E:/GitHubBlog/public/archives/2020/09/index.html","9c41a977b569515cbb41db7a306140e1"],["E:/GitHubBlog/public/archives/2020/10/index.html","7764aac9ef4ddfab711feb6c3ca69323"],["E:/GitHubBlog/public/archives/2020/11/index.html","79ae6173f3238f7431e07a84091eac5e"],["E:/GitHubBlog/public/archives/2020/12/index.html","ddef1ccaacc31e69d31daacfe0abd565"],["E:/GitHubBlog/public/archives/2020/index.html","678ad4dc906e07d3a9a72abf75305dc5"],["E:/GitHubBlog/public/archives/2020/page/2/index.html","30c0edf696901e7884d53c82a49f927e"],["E:/GitHubBlog/public/archives/2020/page/3/index.html","9f8dc8ac6e4056c5a39abacc48dce5e9"],["E:/GitHubBlog/public/archives/2020/page/4/index.html","06a2527960d66a0315343ff8b21615cd"],["E:/GitHubBlog/public/archives/2020/page/5/index.html","ffe212ab3dda6a46e223f07c64e73461"],["E:/GitHubBlog/public/archives/2021/01/index.html","d867d8b6a131d93d7bf389a67eb6f9bb"],["E:/GitHubBlog/public/archives/2021/index.html","d09861427862a5bae00b3af9adcc3f3c"],["E:/GitHubBlog/public/archives/index.html","e81f0ec9876004c5081b02ccb4e13f23"],["E:/GitHubBlog/public/archives/page/2/index.html","affbed7673dedb3f46924ae2c647a185"],["E:/GitHubBlog/public/archives/page/3/index.html","09192dbc2c6b9bc4a134f40632bc3e94"],["E:/GitHubBlog/public/archives/page/4/index.html","209103891a8af160ec269c05ed139aa9"],["E:/GitHubBlog/public/archives/page/5/index.html","38de4ce5438220e2eccda84e4a655f2b"],["E:/GitHubBlog/public/archives/page/6/index.html","ec4e9fdc7d4f99b8b4825c6df2fb7de3"],["E:/GitHubBlog/public/assets/css/APlayer.min.css","fbe994054426fadb2dff69d824c5c67a"],["E:/GitHubBlog/public/assets/js/APlayer.min.js","8f1017e7a73737e631ff95fa51e4e7d7"],["E:/GitHubBlog/public/assets/js/Meting.min.js","bfac0368480fd344282ec018d28f173d"],["E:/GitHubBlog/public/css/404.css","b1bb50e1fabe41adcec483f6427fb3aa"],["E:/GitHubBlog/public/css/index.css","2389364df340948a5b822422956c2c87"],["E:/GitHubBlog/public/css/var.css","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/img/algolia.svg","fd40b88ac5370a5353a50b8175c1f367"],["E:/GitHubBlog/public/img/avatar.png","6cc4a809d23e3d8946a299ae4ce4e4cd"],["E:/GitHubBlog/public/index.html","50f6870ceb2c5baadca75c4790faea3c"],["E:/GitHubBlog/public/js/copy.js","45aae54bf2e43ac27ecc41eb5e0bacf7"],["E:/GitHubBlog/public/js/fancybox.js","f84d626654b9bbc05720952b3effe062"],["E:/GitHubBlog/public/js/fireworks.js","35933ce61c158ef9c33b5e089fe757ac"],["E:/GitHubBlog/public/js/head.js","347edd99f8e3921b45fa617e839d8182"],["E:/GitHubBlog/public/js/hexo-theme-melody.js","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/js/katex.js","d971ba8b8dee1120ef66744b89cfd2b1"],["E:/GitHubBlog/public/js/scroll.js","a12ad7e37b9e325f8da3d0523d3757c7"],["E:/GitHubBlog/public/js/search/algolia.js","53160985d32d6fd66d98aa0e05b081ac"],["E:/GitHubBlog/public/js/search/local-search.js","1565b508bd866ed6fbd724a3858af5d8"],["E:/GitHubBlog/public/js/sidebar.js","d24db780974e661198eeb2e45f20a28f"],["E:/GitHubBlog/public/js/third-party/anime.min.js","9b4bbe6deb700e1c3606eab732f5eea5"],["E:/GitHubBlog/public/js/third-party/canvas-ribbon.js","09c181544ddff1db701db02ac30453e0"],["E:/GitHubBlog/public/js/third-party/jquery.fancybox.min.js","3c9fa1c1199cd4f874d855ecb1641335"],["E:/GitHubBlog/public/js/third-party/jquery.min.js","c9f5aeeca3ad37bf2aa006139b935f0a"],["E:/GitHubBlog/public/js/third-party/reveal/head.min.js","aad121203010122e05f1766d92385214"],["E:/GitHubBlog/public/js/third-party/reveal/highlight.min.js","44594243bec43813a16371af8fe7e105"],["E:/GitHubBlog/public/js/third-party/reveal/markdown.min.js","7ec4cef5a7fe3f0bf0eb4dc6d7bca114"],["E:/GitHubBlog/public/js/third-party/reveal/marked.min.js","c2a88705e206d71dc21fdc4445349127"],["E:/GitHubBlog/public/js/third-party/reveal/math.min.js","0a278fee2e57c530ab07f7d2d9ea8d96"],["E:/GitHubBlog/public/js/third-party/reveal/notes.min.js","89a0dfae4d706f9c75b317f686c3aa14"],["E:/GitHubBlog/public/js/third-party/reveal/reveal.min.js","8988419d67efb5fe93e291a357e26ec9"],["E:/GitHubBlog/public/js/third-party/reveal/zoom.min.js","9791f96e63e7d534cba2b67d4bda0419"],["E:/GitHubBlog/public/js/third-party/velocity.min.js","64da069aba987ea0512cf610600a56d1"],["E:/GitHubBlog/public/js/third-party/velocity.ui.min.js","c8ca438424a080620f7b2f4ee4b0fff1"],["E:/GitHubBlog/public/js/transition.js","911db4268f0f6621073afcced9e1bfef"],["E:/GitHubBlog/public/js/utils.js","3ff3423d966a1c351e9867813b3f6d36"],["E:/GitHubBlog/public/page/2/index.html","263c0c453f1198af8552c5492a96f4a8"],["E:/GitHubBlog/public/page/3/index.html","1ce585b2495ff17659b94649ed975643"],["E:/GitHubBlog/public/page/4/index.html","ddac1b8ec92314fad3092134b3e5bc3b"],["E:/GitHubBlog/public/page/5/index.html","9fabb77acc5e644a2bba0b46d75da4b7"],["E:/GitHubBlog/public/page/6/index.html","baa00f1a3c1f9e6825ac7d13c93f9585"],["E:/GitHubBlog/public/tags/Android/index.html","b416e65ec9465eb750a68fb64c393bc7"],["E:/GitHubBlog/public/tags/NLP/index.html","469b78cca73319fd2be753aef3c4b24d"],["E:/GitHubBlog/public/tags/NLP/page/2/index.html","2fa351b833665393403c1f68196a05c4"],["E:/GitHubBlog/public/tags/NLP/page/3/index.html","468075c628de6553e7370e34b485147c"],["E:/GitHubBlog/public/tags/R/index.html","9653e1226347facedca1c343e67dca0f"],["E:/GitHubBlog/public/tags/index.html","2d754810c4714dfc41633f97ba6d08d4"],["E:/GitHubBlog/public/tags/java/index.html","0c70b7c392de78add67ed855b1d35fb1"],["E:/GitHubBlog/public/tags/leetcode/index.html","f381c0124f53a2f883a9ae3ed384423d"],["E:/GitHubBlog/public/tags/python/index.html","bfa1bd63cb51ca5063311fee67121361"],["E:/GitHubBlog/public/tags/pytorch/index.html","560af47ca254a5a18b1584a20cf2d9f4"],["E:/GitHubBlog/public/tags/优化方法/index.html","0e422b0f16a1c03691fc25749714e9ba"],["E:/GitHubBlog/public/tags/总结/index.html","076d0b22732e60b202ce46915778551d"],["E:/GitHubBlog/public/tags/总结/page/2/index.html","41f0351dc14459994d887bb80a787540"],["E:/GitHubBlog/public/tags/数据分析/index.html","1d5eeb1a59007533ec4af4de2c83b018"],["E:/GitHubBlog/public/tags/数据挖掘/index.html","b34321250787615d9dc9653afb30d88f"],["E:/GitHubBlog/public/tags/数据结构/index.html","1ae4432d9e898bf88a3a0fdefa7ea92b"],["E:/GitHubBlog/public/tags/机器学习/index.html","4d16e67898bc02489494a3050170e266"],["E:/GitHubBlog/public/tags/深度学习/index.html","bb8ee591a53f2b7f138c62d73b2b384e"],["E:/GitHubBlog/public/tags/爬虫/index.html","20e2f661dc2c73d304769382f6210af1"],["E:/GitHubBlog/public/tags/笔记/index.html","919dc26ae8942a6099a658dc61637a49"],["E:/GitHubBlog/public/tags/论文/index.html","16eb769076071056a8a71299a0d074e1"],["E:/GitHubBlog/public/tags/论文/page/2/index.html","2eff08378020ed051f15d2d784aa080f"],["E:/GitHubBlog/public/tags/论文/page/3/index.html","8d8aca6553a0e0dbdf44857aff5a8ab0"],["E:/GitHubBlog/public/tags/读书笔记/index.html","2bfccf7ee68e5fa766e4db1e71d7cbfd"]];
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







