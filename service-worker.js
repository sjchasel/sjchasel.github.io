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

var precacheConfig = [["E:/GitHubBlog/public/2020/01/28/hello-world/index.html","7d032e87538f663c7286c920e9ac9c25"],["E:/GitHubBlog/public/2020/02/08/python爬虫学习总结/index.html","a0b39fde271c98016fa7aed1ca5a492a"],["E:/GitHubBlog/public/2020/02/19/数据结构第一周-绪论/index.html","9200ff3b6677fb5382700b9ad68a93fa"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-1/index.html","7449f87485ad60a8f96aafd91fda28f0"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-2-创建数据集/index.html","346cf225d10edac80fe3190a6bdbfcf7"],["E:/GitHubBlog/public/2020/03/01/《R语言实战》学习总结-3-图形初阶/index.html","b3b483a55277c3d88ba4bedb5bb1aaaa"],["E:/GitHubBlog/public/2020/03/02/数据结构第二周-线性结构1/index.html","e9aac5041b32d4a21fb66ee3c1098da0"],["E:/GitHubBlog/public/2020/03/03/《R语言实战》学习总结-4-基本数据管理/index.html","6ddc0a47d4c9bf25b0634a77f89bf166"],["E:/GitHubBlog/public/2020/03/14/数据结构第三周-线性结构2/index.html","2ad9c1de740cb883a5ae8576b6a65a78"],["E:/GitHubBlog/public/2020/03/14/数据结构第四周-栈与队列/index.html","9e95e2916e996b9ce195fdc11d2f7c34"],["E:/GitHubBlog/public/2020/03/17/leetcode记录-1、88、20/index.html","25f87f0b4bf58f47c5c60990ed0dde4b"],["E:/GitHubBlog/public/2020/03/20/温度转换程序/index.html","7799e89029771fe143c2fdacdfd87bae"],["E:/GitHubBlog/public/2020/03/22/数据结构第五周-递归/index.html","de0a0d07cafb8e5b34c79f05774f2cd6"],["E:/GitHubBlog/public/2020/03/22/数据结构第六周-串与数组/index.html","0744692965af3c13751a61155cf47390"],["E:/GitHubBlog/public/2020/04/02/leetcode记录-52/index.html","818147530dd7b54638503192acb98795"],["E:/GitHubBlog/public/2020/04/17/Android第八周作业-每天第一次打开应用更新数据/index.html","de697d1fa9c214fd7cb6301a526fc98e"],["E:/GitHubBlog/public/2020/05/21/attention模型/index.html","68cc207ae71e98f2d0b3bfb53b4107a7"],["E:/GitHubBlog/public/2020/05/21/编码器与解码器框架/index.html","3d614f9303c22cfa62115f974a24a8c9"],["E:/GitHubBlog/public/2020/07/11/思考，快与慢/index.html","3ef5b5dd0ab64c69383b762649fde021"],["E:/GitHubBlog/public/2020/07/13/20200713-0717总结/index.html","5fe17408e3c732602c6ff494cc47b917"],["E:/GitHubBlog/public/2020/07/13/ZeroShotCeres Zero-Shot Relation Extraction/index.html","32ddd271e7e79ae48ad6a0c2015596aa"],["E:/GitHubBlog/public/2020/07/20/Keyphrase Generation A Text Summarization Struggle/index.html","eb74bd67ebfff2e860e5ea6c21ea6e07"],["E:/GitHubBlog/public/2020/07/20/TPR/index.html","a470ea4938b121235af954d35a0f1e33"],["E:/GitHubBlog/public/2020/07/23/Keyphrase Generation for Scientific Document Retrieval/index.html","ce9c09ec330503a80b344d5782439d96"],["E:/GitHubBlog/public/2020/07/24/20200718-0724总结/index.html","d3bfbdc4422fc0bccdc7fbc27dec4479"],["E:/GitHubBlog/public/2020/07/31/20200725-0731总结/index.html","a8b2174cc9c2aa7f2ab5762ab9024377"],["E:/GitHubBlog/public/2020/08/07/20200801-0807总结/index.html","ae71334fd2a2bda18a0893eeaa2b3ccb"],["E:/GitHubBlog/public/2020/08/12/逻辑回归算法/index.html","2b8b7013998e455332b9acf1b90a310e"],["E:/GitHubBlog/public/2020/08/13/李宏毅-机器学习21/index.html","923cbe003c8cd09189e0d88a5383297b"],["E:/GitHubBlog/public/2020/08/14/20200808-0815总结/index.html","eeb1fcf2281ca646ae69d18766de117b"],["E:/GitHubBlog/public/2020/09/03/20200816-0903总结/index.html","fbad090bb37dd39596876cd849ab9228"],["E:/GitHubBlog/public/2020/09/16/20200914-0918总结/index.html","3865639f3a273f57369bdb985e65e5c1"],["E:/GitHubBlog/public/2020/09/25/20200919-0925总结/index.html","b967bd9436f239b9e14e36201164055c"],["E:/GitHubBlog/public/2020/10/06/《Deep Keyphrase Generation》详细解析！/index.html","8bf6de5d8d10ae586625be0d148a8335"],["E:/GitHubBlog/public/2020/10/16/20201008-1017总结/index.html","17106ff3dcf4fd583d180380d91ba18a"],["E:/GitHubBlog/public/2020/10/20/特征选择的几种方法(一)————Filter过滤法/index.html","06d3ae08e06b1e68d6ebb5f9ac84d57c"],["E:/GitHubBlog/public/2020/10/21/PCA/index.html","ea6f1ec23a6c6fc52204e833e57a4d76"],["E:/GitHubBlog/public/2020/10/22/20201018-1023总结/index.html","c800852200dde95e9c1ff513ce604a5e"],["E:/GitHubBlog/public/2020/10/24/20201024-1030/index.html","9afcc38341cd1a23ddd94565c60334c6"],["E:/GitHubBlog/public/2020/10/30/1030讨论记录/index.html","a31abc2f759e66b431346408d6fae58c"],["E:/GitHubBlog/public/2020/11/03/20201103-总结/index.html","cbabeb7df00c4f82205f3987e89abb39"],["E:/GitHubBlog/public/2020/11/05/线性回归与逻辑回归/index.html","7275352374a0e386ef32c9c434f328ab"],["E:/GitHubBlog/public/2020/11/07/决策树/index.html","226a44428aeea8df1530fd32fd9d2607"],["E:/GitHubBlog/public/2020/11/13/20201107-1113总结/index.html","3b19d3ebe898fb1d20897af08d1c4384"],["E:/GitHubBlog/public/2020/11/15/20201114-119总结/index.html","ec3db4b8d6b86fe1dd48de13b68a7139"],["E:/GitHubBlog/public/2020/11/16/手推一个SVM/index.html","3c65bd5f5718c80f200f8b8e7f033c5f"],["E:/GitHubBlog/public/2020/11/27/20201127左右的总结/index.html","28475f6f9020458007d3efe18d065dc5"],["E:/GitHubBlog/public/2020/12/23/第一个深度学习模型（pytorch）/index.html","6794be37ab19c3b588a47ef7adac25d9"],["E:/GitHubBlog/public/2021/01/02/鲍鱼最终版/index.html","8aa58bfcb4ac09cea248db2ec7f56986"],["E:/GitHubBlog/public/2021/01/08/GSA_NumericalExperiment/index.html","608acc485554cd23b822f2c06d7ff348"],["E:/GitHubBlog/public/2021/01/16/20210115-01总结/index.html","e8a7407446d0f563625a76ab0ce1df97"],["E:/GitHubBlog/public/2021/01/17/20200117-0118总结——OpenNMT阅读（二）/index.html","08cd806f1e561e0d20453be1aa065bc6"],["E:/GitHubBlog/public/2021/01/17/OpenNMT源码解析/index.html","ba82085258eb7e818f206dc7027fc799"],["E:/GitHubBlog/public/2021/01/18/OpenNMT源码解析——onmt框架/index.html","1f69f9b74aab170c8b7516afd4145f86"],["E:/GitHubBlog/public/2021/01/19/OpenNMT源码解析modules/index.html","4db978895cc7d56f0066293d1b299f1a"],["E:/GitHubBlog/public/2021/01/19/title复现中的问题/index.html","2a6f9b7e3ceb318f95421fdb1946a946"],["E:/GitHubBlog/public/2021/01/19/重学数据结构——线性表/index.html","8eaf08211e78910226dab6efef5102d0"],["E:/GitHubBlog/public/archives/2020/01/index.html","008266db27f5cc0640913de4d4c71e89"],["E:/GitHubBlog/public/archives/2020/02/index.html","ae676ce1c73bf572d4d77542b5be7467"],["E:/GitHubBlog/public/archives/2020/03/index.html","44e4e36e656860d5b211230b3d552556"],["E:/GitHubBlog/public/archives/2020/04/index.html","47effe1f06407d4651d2594bbed44ad5"],["E:/GitHubBlog/public/archives/2020/05/index.html","72b8fcdc4eee87953d9a77f190ea7200"],["E:/GitHubBlog/public/archives/2020/07/index.html","31e04cc6c9e8e9857ae2bda917a46a1c"],["E:/GitHubBlog/public/archives/2020/08/index.html","675b670ceac5cd14d2f1b17f9de6ef5e"],["E:/GitHubBlog/public/archives/2020/09/index.html","ef619291b444f8a23d0494a47dd5e795"],["E:/GitHubBlog/public/archives/2020/10/index.html","34b3f615ea4dc56ae0c85f936695ff42"],["E:/GitHubBlog/public/archives/2020/11/index.html","71317a356465a13fd834335fdb436828"],["E:/GitHubBlog/public/archives/2020/12/index.html","7ca69510d2a7ece67d090045526bef3c"],["E:/GitHubBlog/public/archives/2020/index.html","6cd4e7e77a77840daf61dcd26659e166"],["E:/GitHubBlog/public/archives/2020/page/2/index.html","410eaca127868eed5b35a0253b71ec90"],["E:/GitHubBlog/public/archives/2020/page/3/index.html","0b14ab290086b44f0cfd4eeae70ed72b"],["E:/GitHubBlog/public/archives/2020/page/4/index.html","b2995213823d109bc65e72da781ffcd3"],["E:/GitHubBlog/public/archives/2020/page/5/index.html","d800590cf50f35801da33fd8a8072505"],["E:/GitHubBlog/public/archives/2021/01/index.html","7c759914e5c8d6a143236fba0fd25255"],["E:/GitHubBlog/public/archives/2021/index.html","93ad793efc02f7794d0f9264d3935818"],["E:/GitHubBlog/public/archives/index.html","b5ab12c2b91583ff9eb04ec614944323"],["E:/GitHubBlog/public/archives/page/2/index.html","a80c429851d630ce1ab7416532c3fded"],["E:/GitHubBlog/public/archives/page/3/index.html","be56dbb285644398fee70f4e9f9c5c6e"],["E:/GitHubBlog/public/archives/page/4/index.html","5e0ad3b15b4eea63de1b84be590db7aa"],["E:/GitHubBlog/public/archives/page/5/index.html","dddab1e61e2f25519adf2e9a43b1399a"],["E:/GitHubBlog/public/archives/page/6/index.html","d5d4e8b26352ff152a222981c3e98853"],["E:/GitHubBlog/public/assets/css/APlayer.min.css","fbe994054426fadb2dff69d824c5c67a"],["E:/GitHubBlog/public/assets/js/APlayer.min.js","8f1017e7a73737e631ff95fa51e4e7d7"],["E:/GitHubBlog/public/assets/js/Meting.min.js","bfac0368480fd344282ec018d28f173d"],["E:/GitHubBlog/public/css/404.css","b1bb50e1fabe41adcec483f6427fb3aa"],["E:/GitHubBlog/public/css/index.css","2389364df340948a5b822422956c2c87"],["E:/GitHubBlog/public/css/var.css","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/img/algolia.svg","fd40b88ac5370a5353a50b8175c1f367"],["E:/GitHubBlog/public/img/avatar.png","6cc4a809d23e3d8946a299ae4ce4e4cd"],["E:/GitHubBlog/public/index.html","17b4609e86766301f8627f188d256322"],["E:/GitHubBlog/public/js/copy.js","45aae54bf2e43ac27ecc41eb5e0bacf7"],["E:/GitHubBlog/public/js/fancybox.js","f84d626654b9bbc05720952b3effe062"],["E:/GitHubBlog/public/js/fireworks.js","35933ce61c158ef9c33b5e089fe757ac"],["E:/GitHubBlog/public/js/head.js","347edd99f8e3921b45fa617e839d8182"],["E:/GitHubBlog/public/js/hexo-theme-melody.js","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/js/katex.js","d971ba8b8dee1120ef66744b89cfd2b1"],["E:/GitHubBlog/public/js/scroll.js","a12ad7e37b9e325f8da3d0523d3757c7"],["E:/GitHubBlog/public/js/search/algolia.js","53160985d32d6fd66d98aa0e05b081ac"],["E:/GitHubBlog/public/js/search/local-search.js","1565b508bd866ed6fbd724a3858af5d8"],["E:/GitHubBlog/public/js/sidebar.js","d24db780974e661198eeb2e45f20a28f"],["E:/GitHubBlog/public/js/third-party/anime.min.js","9b4bbe6deb700e1c3606eab732f5eea5"],["E:/GitHubBlog/public/js/third-party/canvas-ribbon.js","09c181544ddff1db701db02ac30453e0"],["E:/GitHubBlog/public/js/third-party/jquery.fancybox.min.js","3c9fa1c1199cd4f874d855ecb1641335"],["E:/GitHubBlog/public/js/third-party/jquery.min.js","c9f5aeeca3ad37bf2aa006139b935f0a"],["E:/GitHubBlog/public/js/third-party/reveal/head.min.js","aad121203010122e05f1766d92385214"],["E:/GitHubBlog/public/js/third-party/reveal/highlight.min.js","44594243bec43813a16371af8fe7e105"],["E:/GitHubBlog/public/js/third-party/reveal/markdown.min.js","7ec4cef5a7fe3f0bf0eb4dc6d7bca114"],["E:/GitHubBlog/public/js/third-party/reveal/marked.min.js","c2a88705e206d71dc21fdc4445349127"],["E:/GitHubBlog/public/js/third-party/reveal/math.min.js","0a278fee2e57c530ab07f7d2d9ea8d96"],["E:/GitHubBlog/public/js/third-party/reveal/notes.min.js","89a0dfae4d706f9c75b317f686c3aa14"],["E:/GitHubBlog/public/js/third-party/reveal/reveal.min.js","8988419d67efb5fe93e291a357e26ec9"],["E:/GitHubBlog/public/js/third-party/reveal/zoom.min.js","9791f96e63e7d534cba2b67d4bda0419"],["E:/GitHubBlog/public/js/third-party/velocity.min.js","64da069aba987ea0512cf610600a56d1"],["E:/GitHubBlog/public/js/third-party/velocity.ui.min.js","c8ca438424a080620f7b2f4ee4b0fff1"],["E:/GitHubBlog/public/js/transition.js","911db4268f0f6621073afcced9e1bfef"],["E:/GitHubBlog/public/js/utils.js","3ff3423d966a1c351e9867813b3f6d36"],["E:/GitHubBlog/public/page/2/index.html","6c68d426abe103c3c843aa0b5c767371"],["E:/GitHubBlog/public/page/3/index.html","c141092b06c10068b190896fe6526b2e"],["E:/GitHubBlog/public/page/4/index.html","7a6fec927625b16a2a3f2989b2b7492b"],["E:/GitHubBlog/public/page/5/index.html","0be97e69673669586f7d8393b1f77f44"],["E:/GitHubBlog/public/page/6/index.html","a5c63bc759ab6372f769234a41c9546c"],["E:/GitHubBlog/public/tags/Android/index.html","fde9a360db39f78fc5ff2ba7dd54b4ed"],["E:/GitHubBlog/public/tags/NLP/index.html","ce38165c34f97d35775b5f2060f08a23"],["E:/GitHubBlog/public/tags/NLP/page/2/index.html","463b84f6add1af96b44190772644637b"],["E:/GitHubBlog/public/tags/NLP/page/3/index.html","ff0fb1b35371b306e261fe1cf4318753"],["E:/GitHubBlog/public/tags/R/index.html","2a88e03edba64676c7b516a22d62a2fb"],["E:/GitHubBlog/public/tags/index.html","7732572caded51953b1cc09b55352453"],["E:/GitHubBlog/public/tags/java/index.html","faf50e719ae7eeb3a4f024fa7c734214"],["E:/GitHubBlog/public/tags/java/page/2/index.html","a6657ac425b2528af5807908eed7285a"],["E:/GitHubBlog/public/tags/leetcode/index.html","966e1f26a24c898a081907b3ab9ba43d"],["E:/GitHubBlog/public/tags/python/index.html","200c09286f8b312ee9fdf26498e7e668"],["E:/GitHubBlog/public/tags/pytorch/index.html","cd88933d27b7700cf56e97aa98ad74e3"],["E:/GitHubBlog/public/tags/优化方法/index.html","2d4460139f1333ebf6004eda75b4be97"],["E:/GitHubBlog/public/tags/总结/index.html","7b4e0fb81328b75a6d8ce2db849f9221"],["E:/GitHubBlog/public/tags/总结/page/2/index.html","f3ea6efd763cab61fa2175bdad80214a"],["E:/GitHubBlog/public/tags/数据分析/index.html","8cdfc3b785f3e90f4119d748ce2368c9"],["E:/GitHubBlog/public/tags/数据挖掘/index.html","f0d5638f1d88047ff1914dc661be13b6"],["E:/GitHubBlog/public/tags/数据结构/index.html","370c15d1dec57e66f0630c9c15849a63"],["E:/GitHubBlog/public/tags/机器学习/index.html","e6be1cd2831fd8c9ac91b7807299279b"],["E:/GitHubBlog/public/tags/深度学习/index.html","537fc8ad7c1f436c90182df23d9e68d4"],["E:/GitHubBlog/public/tags/爬虫/index.html","48216bc3bef5f0c2e2b38f73128057fb"],["E:/GitHubBlog/public/tags/笔记/index.html","9af9b1eb2ef106843aff76059b7a8bc3"],["E:/GitHubBlog/public/tags/论文/index.html","1906228bb1f1b0856f193ded5e6a72c8"],["E:/GitHubBlog/public/tags/论文/page/2/index.html","d3387b219626c38d16c8574269930527"],["E:/GitHubBlog/public/tags/论文/page/3/index.html","c6bf7cf2c88a81b6f4eb3bec2194336a"],["E:/GitHubBlog/public/tags/读书笔记/index.html","5de09391b8a145006a7477c6cad8c089"]];
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







