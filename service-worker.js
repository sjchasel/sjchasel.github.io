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

var precacheConfig = [["E:/GitHubBlog/public/2020/01/28/hello-world/index.html","592ef9898cf82dd0ccb5ebb8e2681a4b"],["E:/GitHubBlog/public/2020/02/08/python爬虫学习总结/index.html","078475dce97f63fb7cf5d5f3f58b72cf"],["E:/GitHubBlog/public/2020/02/19/数据结构第一周-绪论/index.html","c5805677ba6c171e033a7a4f383284c2"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-1/index.html","bdf8a54fb1a7c4375717b0cb0199a027"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-2-创建数据集/index.html","4ee4386bffd7f4f396539038af9f2a88"],["E:/GitHubBlog/public/2020/03/01/《R语言实战》学习总结-3-图形初阶/index.html","b879ccaddbb62e3ab63f6579a27b30d6"],["E:/GitHubBlog/public/2020/03/02/数据结构第二周-线性结构1/index.html","067b56f6e164f9304a6c1a06743e3971"],["E:/GitHubBlog/public/2020/03/03/《R语言实战》学习总结-4-基本数据管理/index.html","7b8ed71bd4e7534b220e58a7adffc943"],["E:/GitHubBlog/public/2020/03/14/数据结构第三周-线性结构2/index.html","affa98efe6a906f4ea9c115d4eeefd82"],["E:/GitHubBlog/public/2020/03/14/数据结构第四周-栈与队列/index.html","18ff762e100d7ac54048bf69af071bb8"],["E:/GitHubBlog/public/2020/03/17/leetcode记录-1、88、20/index.html","385eaca5ed19a2848ba2254e4bd8cf8e"],["E:/GitHubBlog/public/2020/03/20/温度转换程序/index.html","8806a7ced76c15162e5850f4eba35883"],["E:/GitHubBlog/public/2020/03/22/数据结构第五周-递归/index.html","e7c3b137833fd0ca945a66b1173b492f"],["E:/GitHubBlog/public/2020/03/22/数据结构第六周-串与数组/index.html","bb4f2290267a80ee4f71e6d551e0aa05"],["E:/GitHubBlog/public/2020/04/02/leetcode记录-52/index.html","239850377afa14cddb48fb2d7fdded5f"],["E:/GitHubBlog/public/2020/04/17/Android第八周作业-每天第一次打开应用更新数据/index.html","e8c5a1061cb6d70b91f226229a04a44d"],["E:/GitHubBlog/public/2020/05/21/attention模型/index.html","1f465cef09e80f691b498acc85ac0b05"],["E:/GitHubBlog/public/2020/05/21/编码器与解码器框架/index.html","615b42b7c76e1f9dce16384e2e850522"],["E:/GitHubBlog/public/2020/07/11/思考，快与慢/index.html","850debbc76eb84977c2d25a048224099"],["E:/GitHubBlog/public/2020/07/13/20200713-0717总结/index.html","c2a2ed2883aa28c66d6ac5d784b22e1e"],["E:/GitHubBlog/public/2020/07/13/ZeroShotCeres Zero-Shot Relation Extraction/index.html","42873c1ca3ac1bc38e5a02a44b2e3675"],["E:/GitHubBlog/public/2020/07/20/Keyphrase Generation A Text Summarization Struggle/index.html","204a467c520914d11741b0dfa9d34f11"],["E:/GitHubBlog/public/2020/07/20/TPR/index.html","48a04c36e16c75f23e5498ad5e05a89d"],["E:/GitHubBlog/public/2020/07/23/Keyphrase Generation for Scientific Document Retrieval/index.html","584533421886aada6f8a6352daaf5f65"],["E:/GitHubBlog/public/2020/07/24/20200718-0724总结/index.html","08d7dd8344a9f288412811504aa73660"],["E:/GitHubBlog/public/2020/07/31/20200725-0731总结/index.html","9036610aa2803492702bb2c406542381"],["E:/GitHubBlog/public/2020/08/07/20200801-0807总结/index.html","2d6e833f5139fec17e68d29ac3b7d705"],["E:/GitHubBlog/public/2020/08/12/逻辑回归算法/index.html","14acd78d30ff0d9973e8ebea9e02f882"],["E:/GitHubBlog/public/2020/08/13/李宏毅-机器学习21/index.html","0fe6f6e3440b480173b0bff3065671c3"],["E:/GitHubBlog/public/2020/08/14/20200808-0815总结/index.html","6fb6ea749611dd435a87c96803e4c050"],["E:/GitHubBlog/public/2020/09/03/20200816-0903总结/index.html","09a40f406b87a20dd3b1ad6e8047fe74"],["E:/GitHubBlog/public/2020/09/16/20200914-0918总结/index.html","51a6ab9083024514886df421517b20ad"],["E:/GitHubBlog/public/2020/09/25/20200919-0925总结/index.html","e9f9551d1b514248e179fb52ea78da4d"],["E:/GitHubBlog/public/2020/10/06/《Deep Keyphrase Generation》详细解析！/index.html","aae823b738f675a60a861aa5ee04ef94"],["E:/GitHubBlog/public/2020/10/16/20201008-1017总结/index.html","ef4eabe0ca177fecf47c38145e1070ea"],["E:/GitHubBlog/public/2020/10/20/特征选择的几种方法(一)————Filter过滤法/index.html","1573f373abd7cac8b7c26badb8a7b86c"],["E:/GitHubBlog/public/2020/10/21/PCA/index.html","3c612787ac6f0bf6d39cffb078e3e7a4"],["E:/GitHubBlog/public/2020/10/22/20201018-1023总结/index.html","f4b0b463d12fe43b5eec3824f4a9c8b4"],["E:/GitHubBlog/public/2020/10/24/20201024-1030/index.html","aa76d88e33ecc600ee0399b29a35d2e7"],["E:/GitHubBlog/public/2020/10/30/1030讨论记录/index.html","0248db5bc12c151663b4391dad1c1426"],["E:/GitHubBlog/public/2020/11/03/20201103-总结/index.html","fc4f93a5e1f6449b17b1dc6c84a1bef1"],["E:/GitHubBlog/public/2020/11/05/线性回归与逻辑回归/index.html","fa51349fc9a783c31f8a076d4a6af718"],["E:/GitHubBlog/public/2020/11/07/决策树/index.html","636f5a9d19eda538d309888d2069ba03"],["E:/GitHubBlog/public/2020/11/13/20201107-1113总结/index.html","08ac29ce7f5f3b56941e4ba1973a2935"],["E:/GitHubBlog/public/2020/11/15/20201114-119总结/index.html","e0c6b346ab464c36a316a1d69dee30cd"],["E:/GitHubBlog/public/2020/11/16/手推一个SVM/index.html","4ff9fec551796fbe3cb43499b30b7f74"],["E:/GitHubBlog/public/2020/11/27/20201127左右的总结/index.html","ab225379f11970962da5db5b0a844942"],["E:/GitHubBlog/public/2020/12/23/第一个深度学习模型（pytorch）/index.html","4b8e6875b0c8cf4e1df283dbb08e871e"],["E:/GitHubBlog/public/2021/01/02/鲍鱼最终版/index.html","c8af49e2360e09e254ddb1382ae4761c"],["E:/GitHubBlog/public/2021/01/08/GSA_NumericalExperiment/index.html","adc9e6f2533df4a35a67fe401f2a109c"],["E:/GitHubBlog/public/2021/01/16/20210115-01总结/index.html","fd24da8e112b96620f3d7337e8e3c452"],["E:/GitHubBlog/public/2021/01/17/20200117-0118总结——OpenNMT阅读（二）/index.html","077d1a3be44216b47e353fdcdb7cc0b5"],["E:/GitHubBlog/public/2021/01/17/OpenNMT源码解析/index.html","0f493741359c4395dbc8fa6e2101e2b7"],["E:/GitHubBlog/public/2021/01/19/重学数据结构——线性表/index.html","5c0d217940695d6926e5e92f660a7b40"],["E:/GitHubBlog/public/2021/01/22/《深度学习入门-基于python的理论与实现》——第二章 感知机/index.html","2a2201e5d82dcd6640f7d54a65cd155e"],["E:/GitHubBlog/public/2021/01/22/代码理论/index.html","b2db6b7683ec123bcc246ef9443c054e"],["E:/GitHubBlog/public/2021/01/26/title复现中的问题/index.html","0ce5601c8036279581e0c42938214d18"],["E:/GitHubBlog/public/2021/01/26/简单链表题/index.html","c4481a5ecde058f8c01d2a753f29d016"],["E:/GitHubBlog/public/archives/2020/01/index.html","588fc729292f56739f65e81d685a1a51"],["E:/GitHubBlog/public/archives/2020/02/index.html","8c7f9183b0dee85918f3f702862ceae2"],["E:/GitHubBlog/public/archives/2020/03/index.html","d0fe4347abfbee02ccd90778399fbb0e"],["E:/GitHubBlog/public/archives/2020/04/index.html","8531ba52e5b98e25f83d50a8efa65777"],["E:/GitHubBlog/public/archives/2020/05/index.html","a50e03b45a9065ea785115f5a683bbb1"],["E:/GitHubBlog/public/archives/2020/07/index.html","867011dfc88dffc27bf2e2c207de9202"],["E:/GitHubBlog/public/archives/2020/08/index.html","5809e596494b138143ec729ac3d201b6"],["E:/GitHubBlog/public/archives/2020/09/index.html","e845452f29c5e7fe19c1f3bfce7b81a1"],["E:/GitHubBlog/public/archives/2020/10/index.html","2e18efd4923c653730862227c4606219"],["E:/GitHubBlog/public/archives/2020/11/index.html","3a555bee49aa2f7f7d459578438c21ea"],["E:/GitHubBlog/public/archives/2020/12/index.html","6ba9c96adf9997cbf9bd8324bfee0df4"],["E:/GitHubBlog/public/archives/2020/index.html","7e2139725cac0af548b7d80971aeb0b4"],["E:/GitHubBlog/public/archives/2020/page/2/index.html","b1bbb955d53e7737831aff18c0c6f805"],["E:/GitHubBlog/public/archives/2020/page/3/index.html","e0593f87e59f69546dcaa6d249aff821"],["E:/GitHubBlog/public/archives/2020/page/4/index.html","06d805e31ad3303babba7098e4a37355"],["E:/GitHubBlog/public/archives/2020/page/5/index.html","f2d0cca127c82b83c86d95f742def42d"],["E:/GitHubBlog/public/archives/2021/01/index.html","b91b11362e1d09cdf7cf8d08a510d409"],["E:/GitHubBlog/public/archives/2021/index.html","0b9a2794d86122b872fdc64e3363dc73"],["E:/GitHubBlog/public/archives/index.html","0f1782aa7c1bf3074fbb32cca9315c78"],["E:/GitHubBlog/public/archives/page/2/index.html","44c2579ffae601832adbbb8e28cd2707"],["E:/GitHubBlog/public/archives/page/3/index.html","2920cde3d8be998b9ca480629d64093a"],["E:/GitHubBlog/public/archives/page/4/index.html","24f12e206218fc6c7ed963b4cd12cf85"],["E:/GitHubBlog/public/archives/page/5/index.html","7a52454d72cd0f7eb7b313754419023a"],["E:/GitHubBlog/public/archives/page/6/index.html","f1d537adfb34cee45c942500fec8b0b0"],["E:/GitHubBlog/public/assets/css/APlayer.min.css","fbe994054426fadb2dff69d824c5c67a"],["E:/GitHubBlog/public/assets/js/APlayer.min.js","8f1017e7a73737e631ff95fa51e4e7d7"],["E:/GitHubBlog/public/assets/js/Meting.min.js","bfac0368480fd344282ec018d28f173d"],["E:/GitHubBlog/public/css/404.css","b1bb50e1fabe41adcec483f6427fb3aa"],["E:/GitHubBlog/public/css/index.css","2389364df340948a5b822422956c2c87"],["E:/GitHubBlog/public/css/var.css","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/img/algolia.svg","fd40b88ac5370a5353a50b8175c1f367"],["E:/GitHubBlog/public/img/avatar.png","6cc4a809d23e3d8946a299ae4ce4e4cd"],["E:/GitHubBlog/public/index.html","d1d51090d686d49960cbfd71ad198e59"],["E:/GitHubBlog/public/js/copy.js","45aae54bf2e43ac27ecc41eb5e0bacf7"],["E:/GitHubBlog/public/js/fancybox.js","f84d626654b9bbc05720952b3effe062"],["E:/GitHubBlog/public/js/fireworks.js","35933ce61c158ef9c33b5e089fe757ac"],["E:/GitHubBlog/public/js/head.js","347edd99f8e3921b45fa617e839d8182"],["E:/GitHubBlog/public/js/hexo-theme-melody.js","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/js/katex.js","d971ba8b8dee1120ef66744b89cfd2b1"],["E:/GitHubBlog/public/js/scroll.js","a12ad7e37b9e325f8da3d0523d3757c7"],["E:/GitHubBlog/public/js/search/algolia.js","53160985d32d6fd66d98aa0e05b081ac"],["E:/GitHubBlog/public/js/search/local-search.js","1565b508bd866ed6fbd724a3858af5d8"],["E:/GitHubBlog/public/js/sidebar.js","d24db780974e661198eeb2e45f20a28f"],["E:/GitHubBlog/public/js/third-party/anime.min.js","9b4bbe6deb700e1c3606eab732f5eea5"],["E:/GitHubBlog/public/js/third-party/canvas-ribbon.js","09c181544ddff1db701db02ac30453e0"],["E:/GitHubBlog/public/js/third-party/jquery.fancybox.min.js","3c9fa1c1199cd4f874d855ecb1641335"],["E:/GitHubBlog/public/js/third-party/jquery.min.js","c9f5aeeca3ad37bf2aa006139b935f0a"],["E:/GitHubBlog/public/js/third-party/reveal/head.min.js","aad121203010122e05f1766d92385214"],["E:/GitHubBlog/public/js/third-party/reveal/highlight.min.js","44594243bec43813a16371af8fe7e105"],["E:/GitHubBlog/public/js/third-party/reveal/markdown.min.js","7ec4cef5a7fe3f0bf0eb4dc6d7bca114"],["E:/GitHubBlog/public/js/third-party/reveal/marked.min.js","c2a88705e206d71dc21fdc4445349127"],["E:/GitHubBlog/public/js/third-party/reveal/math.min.js","0a278fee2e57c530ab07f7d2d9ea8d96"],["E:/GitHubBlog/public/js/third-party/reveal/notes.min.js","89a0dfae4d706f9c75b317f686c3aa14"],["E:/GitHubBlog/public/js/third-party/reveal/reveal.min.js","8988419d67efb5fe93e291a357e26ec9"],["E:/GitHubBlog/public/js/third-party/reveal/zoom.min.js","9791f96e63e7d534cba2b67d4bda0419"],["E:/GitHubBlog/public/js/third-party/velocity.min.js","64da069aba987ea0512cf610600a56d1"],["E:/GitHubBlog/public/js/third-party/velocity.ui.min.js","c8ca438424a080620f7b2f4ee4b0fff1"],["E:/GitHubBlog/public/js/transition.js","911db4268f0f6621073afcced9e1bfef"],["E:/GitHubBlog/public/js/utils.js","3ff3423d966a1c351e9867813b3f6d36"],["E:/GitHubBlog/public/page/2/index.html","179e9c8fef4a280552c935fa3ffb626b"],["E:/GitHubBlog/public/page/3/index.html","5511305385cabb250e1183b194c89f23"],["E:/GitHubBlog/public/page/4/index.html","650bae6518cce64d1c87c0d8bbf86414"],["E:/GitHubBlog/public/page/5/index.html","b55b564078cfcfe7d07b98f9864994db"],["E:/GitHubBlog/public/page/6/index.html","554cf728ef1241273a4e56e844748df8"],["E:/GitHubBlog/public/tags/Android/index.html","e5ce9ad0dff7811e34700bf76f3c6fcf"],["E:/GitHubBlog/public/tags/NLP/index.html","343dee8b7356cc78fdd8cc7a617fe2f3"],["E:/GitHubBlog/public/tags/NLP/page/2/index.html","22a68fc3296ed81947dfa9008dfbeb28"],["E:/GitHubBlog/public/tags/NLP/page/3/index.html","877124e0e7317ccb582b5c725e58480f"],["E:/GitHubBlog/public/tags/R/index.html","1e935c2160612d6450269b635889e46e"],["E:/GitHubBlog/public/tags/index.html","a654e56bbb8e331f2791108adaf3bf42"],["E:/GitHubBlog/public/tags/java/index.html","d138308cfcca56c04e971e38ceca51a7"],["E:/GitHubBlog/public/tags/java/page/2/index.html","d79ad69e73391cb2bd2a0bcf3b59b915"],["E:/GitHubBlog/public/tags/leetcode/index.html","b82cbdca7f2fc9ebb309788e3ce9967a"],["E:/GitHubBlog/public/tags/python/index.html","c388297089ec1e03d9368c8791cb7477"],["E:/GitHubBlog/public/tags/pytorch/index.html","6acbd75b0d747c777ae9d9f9cf61c62c"],["E:/GitHubBlog/public/tags/优化方法/index.html","e76a84bf89f9a139ef1e8ec23824ad50"],["E:/GitHubBlog/public/tags/总结/index.html","3f14c67834994fca6555a9fb73ee1c2e"],["E:/GitHubBlog/public/tags/总结/page/2/index.html","51c69a561c4c2e3566529e42a2d52254"],["E:/GitHubBlog/public/tags/数据分析/index.html","29180aff181f3d8f1d00da6e3022e58f"],["E:/GitHubBlog/public/tags/数据挖掘/index.html","1eb0f45361ddbf0cbe2facbf0327db9a"],["E:/GitHubBlog/public/tags/数据结构/index.html","3be9d703c749d0c43c13c96e483cd9c3"],["E:/GitHubBlog/public/tags/机器学习/index.html","dc856bbd3afcecbd5ad798cf22b4ce40"],["E:/GitHubBlog/public/tags/深度学习/index.html","01871b157c1dc69dcdeb012d4835aab4"],["E:/GitHubBlog/public/tags/爬虫/index.html","1348f526a1db56e54829388f9b4d3a81"],["E:/GitHubBlog/public/tags/笔记/index.html","b330bbef728b6aacb9ebcc5f7dfdaebb"],["E:/GitHubBlog/public/tags/论文/index.html","ef9f99191f03ada8c2ad588dff5358b2"],["E:/GitHubBlog/public/tags/论文/page/2/index.html","3a9991ec8d948d62ca320d995e52d30e"],["E:/GitHubBlog/public/tags/论文/page/3/index.html","6c148f1fc9a4410d484bd60cfec7bfef"],["E:/GitHubBlog/public/tags/读书笔记/index.html","65f7a999d5ba5be2b0f4fa98f8a6f072"]];
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







