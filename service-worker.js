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

var precacheConfig = [["E:/GitHubBlog/public/2020/01/28/hello-world/index.html","0fdee5d34f804274a4e32203c4e9992c"],["E:/GitHubBlog/public/2020/02/08/python爬虫学习总结/index.html","2b6bee1701f8033a67285fdc92a095e5"],["E:/GitHubBlog/public/2020/02/19/数据结构第一周-绪论/index.html","ab831561cad9383ef3a0b6d2af9cc1b1"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-1/index.html","b1c387ec367b69191e901a452b036c83"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-2-创建数据集/index.html","40b947d1a0386c5feb2ca1f3edaafa3b"],["E:/GitHubBlog/public/2020/03/01/《R语言实战》学习总结-3-图形初阶/index.html","590a4ac91df62cdec8e7ffb74a1bcdaa"],["E:/GitHubBlog/public/2020/03/02/数据结构第二周-线性结构1/index.html","67549d49ba6167917e844e04384c17af"],["E:/GitHubBlog/public/2020/03/03/《R语言实战》学习总结-4-基本数据管理/index.html","8d171ef10f7a837840d41c076403aba0"],["E:/GitHubBlog/public/2020/03/14/数据结构第三周-线性结构2/index.html","d5241db3d52025bd896494387c21c364"],["E:/GitHubBlog/public/2020/03/14/数据结构第四周-栈与队列/index.html","b008a9dbe5053679d7bc622f71ad7565"],["E:/GitHubBlog/public/2020/03/17/leetcode记录-1、88、20/index.html","11a04410a446f76970658bf6a1903605"],["E:/GitHubBlog/public/2020/03/20/温度转换程序/index.html","da4911ec00e79f11018761f46d3f9491"],["E:/GitHubBlog/public/2020/03/22/数据结构第五周-递归/index.html","97f60a6e51f20b6ebce1f764458ab91d"],["E:/GitHubBlog/public/2020/03/22/数据结构第六周-串与数组/index.html","ae4787c4e70dd377102bffdf055fec0e"],["E:/GitHubBlog/public/2020/04/02/leetcode记录-52/index.html","86f38edb3ed94242ca1f48cb0e573ed8"],["E:/GitHubBlog/public/2020/04/17/Android第八周作业-每天第一次打开应用更新数据/index.html","e7bbed157fcb392e7a6c0d7dbd17f8dc"],["E:/GitHubBlog/public/2020/05/21/attention模型/index.html","f4a8e82481a83d274cda61c9dfae48e4"],["E:/GitHubBlog/public/2020/05/21/编码器与解码器框架/index.html","4de39a49fd41ea5e02c8eb7ff35c3cc8"],["E:/GitHubBlog/public/2020/07/11/思考，快与慢/index.html","60326dc2719a91fe3bc9bcf5832876cb"],["E:/GitHubBlog/public/2020/07/13/20200713-0717总结/index.html","f7343edccb2ee9ff7ba06a8c60fc5cfd"],["E:/GitHubBlog/public/2020/07/13/ZeroShotCeres Zero-Shot Relation Extraction/index.html","3d8b99ca8ba3857cffb45d98a67fe252"],["E:/GitHubBlog/public/2020/07/20/Keyphrase Generation A Text Summarization Struggle/index.html","7b603d08366ed0838ceb02f23c274839"],["E:/GitHubBlog/public/2020/07/20/TPR/index.html","cd935e66d92ba8360311d98c72a21abc"],["E:/GitHubBlog/public/2020/07/23/Keyphrase Generation for Scientific Document Retrieval/index.html","25fe766ee9c4eb1f8ed8b1dc8017cceb"],["E:/GitHubBlog/public/2020/07/24/20200718-0724总结/index.html","3219b13c9821ff16c5a0b825a19a7de8"],["E:/GitHubBlog/public/2020/07/31/20200725-0731总结/index.html","d653aa2d6058e13f3dcf44aee2e1ece1"],["E:/GitHubBlog/public/2020/08/07/20200801-0807总结/index.html","4626963a668c96d1e49a661c814e0f16"],["E:/GitHubBlog/public/2020/08/12/逻辑回归算法/index.html","611b1950484696edb807a1a6baa460ce"],["E:/GitHubBlog/public/2020/08/13/李宏毅-机器学习21/index.html","e10ed45574c2e9ea04a218daa0a41aa8"],["E:/GitHubBlog/public/2020/08/14/20200808-0815总结/index.html","0dc9127e7fd46f9b4d1828519b772f2d"],["E:/GitHubBlog/public/2020/09/03/20200816-0903总结/index.html","6d4562e54525cc9b278ded4aa126b8df"],["E:/GitHubBlog/public/2020/09/16/20200914-0918总结/index.html","2f6df037267016f158d44de6145a7baf"],["E:/GitHubBlog/public/2020/09/25/20200919-0925总结/index.html","38c7fbf318cdf7729737a33802fb7230"],["E:/GitHubBlog/public/2020/10/06/《Deep Keyphrase Generation》详细解析！/index.html","28cd1fa648162a9eea26a4f4ea6c5b7b"],["E:/GitHubBlog/public/2020/10/16/20201008-1017总结/index.html","10927943f4539db4f27677ca7da41912"],["E:/GitHubBlog/public/2020/10/20/特征选择的几种方法(一)————Filter过滤法/index.html","26bc4313e58b0c9dc681196310c8ebfd"],["E:/GitHubBlog/public/2020/10/21/PCA/index.html","e2dcf244c09ba512907a93c2a5b302d2"],["E:/GitHubBlog/public/2020/10/22/20201018-1023总结/index.html","cc6930729ed2bcc04ecdbab29b8dd363"],["E:/GitHubBlog/public/2020/10/24/20201024-1030/index.html","842b287e23f79468ba7c52e0f0d12d18"],["E:/GitHubBlog/public/2020/10/30/1030讨论记录/index.html","1eda74011f2d1a23e40fdb6635b2ab89"],["E:/GitHubBlog/public/2020/11/03/20201103-总结/index.html","e31a2e97b78425c2556127e6bd3f0dea"],["E:/GitHubBlog/public/2020/11/05/线性回归与逻辑回归/index.html","756d03ae74f104123c4834bf7eb7e679"],["E:/GitHubBlog/public/2020/11/07/决策树/index.html","7b8cd905e0983cd83f9a5de5487484f8"],["E:/GitHubBlog/public/2020/11/13/20201107-1113总结/index.html","ea3efd4013d77f4864964159792f630f"],["E:/GitHubBlog/public/2020/11/15/20201114-119总结/index.html","e6e4f5920e0bcaf23f05d98f6f58f96b"],["E:/GitHubBlog/public/2020/11/16/手推一个SVM/index.html","bbc7d14dfa396edd34643c157a8c516d"],["E:/GitHubBlog/public/2020/11/27/20201127左右的总结/index.html","73d63b4ad84b52f357efeabbeb458400"],["E:/GitHubBlog/public/2020/12/23/第一个深度学习模型（pytorch）/index.html","7154604b6fb09ea5ab35dc4806cba0db"],["E:/GitHubBlog/public/2021/01/02/鲍鱼最终版/index.html","f4fa6e0d5953a99df3d08fb6aa7be980"],["E:/GitHubBlog/public/2021/01/08/GSA_NumericalExperiment/index.html","22294a4aeacf551a7a76e5b7e845c0dc"],["E:/GitHubBlog/public/2021/01/16/20210115-01总结/index.html","7a0cc80da20a09c7dc60ccde30976cf8"],["E:/GitHubBlog/public/2021/01/17/20200117-0118总结——OpenNMT阅读（二）/index.html","b86e220471907a39719c6931fc98cca3"],["E:/GitHubBlog/public/2021/01/17/OpenNMT源码解析/index.html","dc8485504d1903c28095c067fa751921"],["E:/GitHubBlog/public/2021/01/19/重学数据结构——线性表/index.html","38e568046bb810a6470be58520fc7229"],["E:/GitHubBlog/public/2021/01/22/《深度学习入门-基于python的理论与实现》——第二章 感知机/index.html","c6569c03ff9731cd4921eb0d09dabfea"],["E:/GitHubBlog/public/2021/01/22/代码理论/index.html","70cb504909c2b1f8de3b66df7391de4f"],["E:/GitHubBlog/public/2021/01/26/title复现中的问题/index.html","dd78a420917994dd95b765a9094d603c"],["E:/GitHubBlog/public/2021/01/26/简单链表题/index.html","7bd564056e39c148dd8f10b9ca8e6c96"],["E:/GitHubBlog/public/2021/01/28/DeepKeyphraseGeneration/index.html","896100cb8d8184034aa411325339622c"],["E:/GitHubBlog/public/archives/2020/01/index.html","fe353e6b9acf6888ce90c6d1cc7435bf"],["E:/GitHubBlog/public/archives/2020/02/index.html","734dbf62b7c2edaa0d59416524cbca76"],["E:/GitHubBlog/public/archives/2020/03/index.html","b0a8d1d5d43b2b2fc4a87f862375fac8"],["E:/GitHubBlog/public/archives/2020/04/index.html","a98643be86657f915c8ffe984e7500cc"],["E:/GitHubBlog/public/archives/2020/05/index.html","0dfc035a024962a283bbd65e54f73504"],["E:/GitHubBlog/public/archives/2020/07/index.html","a0dbdb843c45e4a85faf7e8438d79774"],["E:/GitHubBlog/public/archives/2020/08/index.html","1415f7ff86c1301c04e463aeefd784fb"],["E:/GitHubBlog/public/archives/2020/09/index.html","1ff6d27c3418252142cac868a35549ee"],["E:/GitHubBlog/public/archives/2020/10/index.html","54dc93a10adb437de5823154fc72631d"],["E:/GitHubBlog/public/archives/2020/11/index.html","d2b7fc28da8a86f88333d2a4ae59e732"],["E:/GitHubBlog/public/archives/2020/12/index.html","ae8c0e93c58e393d8d91f870ab7528ac"],["E:/GitHubBlog/public/archives/2020/index.html","e82c6e98b6eea83b95aa766e3758b857"],["E:/GitHubBlog/public/archives/2020/page/2/index.html","1834af72f651208226ddc8cf26c61a22"],["E:/GitHubBlog/public/archives/2020/page/3/index.html","c4473fa170bddb75d267d2d10678a0a8"],["E:/GitHubBlog/public/archives/2020/page/4/index.html","fb5cbaa4eb1ebffbd6611b8862ad298a"],["E:/GitHubBlog/public/archives/2020/page/5/index.html","10c1af74ad75035b64c19889f1b95515"],["E:/GitHubBlog/public/archives/2021/01/index.html","1a5b74179ef870eb8e124b5774143651"],["E:/GitHubBlog/public/archives/2021/01/page/2/index.html","7cbf7314146b7a29b57e9df6ee523759"],["E:/GitHubBlog/public/archives/2021/index.html","ce46ea99dfff0d4c01e67da779dfb37f"],["E:/GitHubBlog/public/archives/2021/page/2/index.html","27a91e55b54a73d4d4b977a04fac1420"],["E:/GitHubBlog/public/archives/index.html","e550698c3cbb22a73abfac4cb786a650"],["E:/GitHubBlog/public/archives/page/2/index.html","892ab4694f7c457476ca80bed7c197dd"],["E:/GitHubBlog/public/archives/page/3/index.html","c9d89cf77ab283b705d91eb08aad25d8"],["E:/GitHubBlog/public/archives/page/4/index.html","df9723df6110a70672eda50ecc745744"],["E:/GitHubBlog/public/archives/page/5/index.html","f7f417fd7577052fa4d982816dd44c53"],["E:/GitHubBlog/public/archives/page/6/index.html","de97b45ba762f478e79eb2339a6bc385"],["E:/GitHubBlog/public/assets/css/APlayer.min.css","fbe994054426fadb2dff69d824c5c67a"],["E:/GitHubBlog/public/assets/js/APlayer.min.js","8f1017e7a73737e631ff95fa51e4e7d7"],["E:/GitHubBlog/public/assets/js/Meting.min.js","bfac0368480fd344282ec018d28f173d"],["E:/GitHubBlog/public/css/404.css","b1bb50e1fabe41adcec483f6427fb3aa"],["E:/GitHubBlog/public/css/index.css","2389364df340948a5b822422956c2c87"],["E:/GitHubBlog/public/css/var.css","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/img/algolia.svg","fd40b88ac5370a5353a50b8175c1f367"],["E:/GitHubBlog/public/img/avatar.png","6cc4a809d23e3d8946a299ae4ce4e4cd"],["E:/GitHubBlog/public/index.html","319aa5c48c99fdd0513d53960fb59652"],["E:/GitHubBlog/public/js/copy.js","45aae54bf2e43ac27ecc41eb5e0bacf7"],["E:/GitHubBlog/public/js/fancybox.js","f84d626654b9bbc05720952b3effe062"],["E:/GitHubBlog/public/js/fireworks.js","35933ce61c158ef9c33b5e089fe757ac"],["E:/GitHubBlog/public/js/head.js","347edd99f8e3921b45fa617e839d8182"],["E:/GitHubBlog/public/js/hexo-theme-melody.js","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/js/katex.js","d971ba8b8dee1120ef66744b89cfd2b1"],["E:/GitHubBlog/public/js/scroll.js","a12ad7e37b9e325f8da3d0523d3757c7"],["E:/GitHubBlog/public/js/search/algolia.js","53160985d32d6fd66d98aa0e05b081ac"],["E:/GitHubBlog/public/js/search/local-search.js","1565b508bd866ed6fbd724a3858af5d8"],["E:/GitHubBlog/public/js/sidebar.js","d24db780974e661198eeb2e45f20a28f"],["E:/GitHubBlog/public/js/third-party/anime.min.js","9b4bbe6deb700e1c3606eab732f5eea5"],["E:/GitHubBlog/public/js/third-party/canvas-ribbon.js","09c181544ddff1db701db02ac30453e0"],["E:/GitHubBlog/public/js/third-party/jquery.fancybox.min.js","3c9fa1c1199cd4f874d855ecb1641335"],["E:/GitHubBlog/public/js/third-party/jquery.min.js","c9f5aeeca3ad37bf2aa006139b935f0a"],["E:/GitHubBlog/public/js/third-party/reveal/head.min.js","aad121203010122e05f1766d92385214"],["E:/GitHubBlog/public/js/third-party/reveal/highlight.min.js","44594243bec43813a16371af8fe7e105"],["E:/GitHubBlog/public/js/third-party/reveal/markdown.min.js","7ec4cef5a7fe3f0bf0eb4dc6d7bca114"],["E:/GitHubBlog/public/js/third-party/reveal/marked.min.js","c2a88705e206d71dc21fdc4445349127"],["E:/GitHubBlog/public/js/third-party/reveal/math.min.js","0a278fee2e57c530ab07f7d2d9ea8d96"],["E:/GitHubBlog/public/js/third-party/reveal/notes.min.js","89a0dfae4d706f9c75b317f686c3aa14"],["E:/GitHubBlog/public/js/third-party/reveal/reveal.min.js","8988419d67efb5fe93e291a357e26ec9"],["E:/GitHubBlog/public/js/third-party/reveal/zoom.min.js","9791f96e63e7d534cba2b67d4bda0419"],["E:/GitHubBlog/public/js/third-party/velocity.min.js","64da069aba987ea0512cf610600a56d1"],["E:/GitHubBlog/public/js/third-party/velocity.ui.min.js","c8ca438424a080620f7b2f4ee4b0fff1"],["E:/GitHubBlog/public/js/transition.js","911db4268f0f6621073afcced9e1bfef"],["E:/GitHubBlog/public/js/utils.js","3ff3423d966a1c351e9867813b3f6d36"],["E:/GitHubBlog/public/page/2/index.html","6df0d66f0c208900a22fbb9809ab27a4"],["E:/GitHubBlog/public/page/3/index.html","bbfbc23fd0bd6e9d9681922f435a4585"],["E:/GitHubBlog/public/page/4/index.html","e0fdd94ddd33e1e561ceb47071bf16b3"],["E:/GitHubBlog/public/page/5/index.html","27164ca258309e269bd6e6be06341305"],["E:/GitHubBlog/public/page/6/index.html","f1a9825a3ff1bfbb06c82780851d11f2"],["E:/GitHubBlog/public/tags/Android/index.html","3618f39bf05460f396b6c5ebef716cc1"],["E:/GitHubBlog/public/tags/NLP/index.html","b595ce1c2d448ab5ebef258d55889779"],["E:/GitHubBlog/public/tags/NLP/page/2/index.html","e7611cc16bb2c3a063cb12c67ca25fd1"],["E:/GitHubBlog/public/tags/NLP/page/3/index.html","fcdcdc9ba52213ab476f04eac6a3e3d7"],["E:/GitHubBlog/public/tags/R/index.html","be144c101f38747f7d1b0c519732cfa5"],["E:/GitHubBlog/public/tags/index.html","be345395c1b3e8d5bc4976f5272f2c1f"],["E:/GitHubBlog/public/tags/java/index.html","f0978b4bc9a08e98e3650e3090f69cf6"],["E:/GitHubBlog/public/tags/java/page/2/index.html","b4e64efa8d9190f3f60466201eba078a"],["E:/GitHubBlog/public/tags/leetcode/index.html","e00f4d29f976385ef208b8465271705d"],["E:/GitHubBlog/public/tags/python/index.html","1695ecf6d9a41d9537d37b0bbe7cfc95"],["E:/GitHubBlog/public/tags/pytorch/index.html","c48201b8a893a5f6fecd6839f294f0b9"],["E:/GitHubBlog/public/tags/优化方法/index.html","f7b0e6394bfc4004558ab71a1511d433"],["E:/GitHubBlog/public/tags/总结/index.html","7bd6c0acfab5340f23e92f9588f0148d"],["E:/GitHubBlog/public/tags/总结/page/2/index.html","52486af0cd6978cf18c0f9f313b1cf8b"],["E:/GitHubBlog/public/tags/数据分析/index.html","5bc78d6d2fc52b0fee490980edecf4c5"],["E:/GitHubBlog/public/tags/数据挖掘/index.html","03f550c5bb120411bb0e104ae04b04fe"],["E:/GitHubBlog/public/tags/数据结构/index.html","eb3d00a0825002799da75e50a04e0653"],["E:/GitHubBlog/public/tags/机器学习/index.html","68fc66c7317860944fb3681de709e8f1"],["E:/GitHubBlog/public/tags/深度学习/index.html","5a24fa4443d8961edc0d1897b697a3c7"],["E:/GitHubBlog/public/tags/爬虫/index.html","2ddf1f8f1e52fecbfef8265a584bd214"],["E:/GitHubBlog/public/tags/笔记/index.html","af6de1216b19b1ffaad838452c2a1ffa"],["E:/GitHubBlog/public/tags/论文/index.html","783cfcef798805107b73fadedaf4e7a6"],["E:/GitHubBlog/public/tags/论文/page/2/index.html","ac5c29433bb97dba3185b72ea1d5f284"],["E:/GitHubBlog/public/tags/论文/page/3/index.html","4900919e450d6759f91487956ba596e1"],["E:/GitHubBlog/public/tags/读书笔记/index.html","55fb5983990d865ab825913f0fe3c208"]];
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







