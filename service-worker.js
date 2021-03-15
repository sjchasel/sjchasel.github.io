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

var precacheConfig = [["E:/GitHubBlog/public/2020/01/28/hello-world/index.html","04d051d756cd7642e233d2b9d533ee45"],["E:/GitHubBlog/public/2020/02/08/python爬虫学习总结/index.html","0adf3dffa01c8ffa9b7f9c6367da8e1f"],["E:/GitHubBlog/public/2020/02/19/数据结构第一周-绪论/index.html","fe13dabab16c82f140ce4b7d2e988f66"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-1/index.html","54605022ba274b507a128341b9d9a163"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-2-创建数据集/index.html","acbec92e8cd916a4d86b4cecd780a2fa"],["E:/GitHubBlog/public/2020/03/01/《R语言实战》学习总结-3-图形初阶/index.html","88e11b6b821b390f654e6745316a1af4"],["E:/GitHubBlog/public/2020/03/02/数据结构第二周-线性结构1/index.html","71178fbce4c480ca87fd593eeea4537d"],["E:/GitHubBlog/public/2020/03/03/《R语言实战》学习总结-4-基本数据管理/index.html","7a891cbdc9a99189e4a57d2f8d7ddcfa"],["E:/GitHubBlog/public/2020/03/14/数据结构第三周-线性结构2/index.html","3d11857bb7da2d3d34cea6c78c32ed38"],["E:/GitHubBlog/public/2020/03/14/数据结构第四周-栈与队列/index.html","4b1e41b5bd5d9236a41461e16919ce0b"],["E:/GitHubBlog/public/2020/03/17/leetcode记录-1、88、20/index.html","563623d85c8cd59286bbe9f5dbc3e118"],["E:/GitHubBlog/public/2020/03/20/温度转换程序/index.html","f9c458a9d64157702a008999d088cde8"],["E:/GitHubBlog/public/2020/03/22/数据结构第五周-递归/index.html","b317d058b20e30c2024366c0726ac7bf"],["E:/GitHubBlog/public/2020/03/22/数据结构第六周-串与数组/index.html","e8f2d57b6c16284d559438c4b09403e0"],["E:/GitHubBlog/public/2020/04/02/leetcode记录-52/index.html","4a023fa7a8064d01366b4a02b9bcc6ce"],["E:/GitHubBlog/public/2020/04/17/Android第八周作业-每天第一次打开应用更新数据/index.html","0822ebd786b82589896ce5513d3996e9"],["E:/GitHubBlog/public/2020/05/21/attention模型/index.html","ea7383c15c7760d4ec58e4810e9d0f90"],["E:/GitHubBlog/public/2020/05/21/编码器与解码器框架/index.html","48f13f04fbcb1ac2df10c3a3e0e7a507"],["E:/GitHubBlog/public/2020/07/11/思考，快与慢/index.html","052892d3a2a2ad71dec37f7bf39ba6db"],["E:/GitHubBlog/public/2020/07/13/20200713-0717总结/index.html","24e51328874c08775032cb33602d767d"],["E:/GitHubBlog/public/2020/07/13/ZeroShotCeres Zero-Shot Relation Extraction/index.html","0fc255ddb51522f95c411e68dc51a3e1"],["E:/GitHubBlog/public/2020/07/20/Keyphrase Generation A Text Summarization Struggle/index.html","d2f4b43438f87f321817071ba0f761f0"],["E:/GitHubBlog/public/2020/07/20/TPR/index.html","9c9c9fdfcb9e3f9ee41cdc93dd58a653"],["E:/GitHubBlog/public/2020/07/23/Keyphrase Generation for Scientific Document Retrieval/index.html","692a1993429fd81d04eeb9804af223f3"],["E:/GitHubBlog/public/2020/07/24/20200718-0724总结/index.html","e768b3d16aacc4ccaebeb3cc8890c6eb"],["E:/GitHubBlog/public/2020/07/31/20200725-0731总结/index.html","c899903ad84759a2c008277d125c911d"],["E:/GitHubBlog/public/2020/08/07/20200801-0807总结/index.html","5587c0df1a26b33da01829fbe2355138"],["E:/GitHubBlog/public/2020/08/12/逻辑回归算法/index.html","5abdcdfb85fef16315c5fdd3c280eda6"],["E:/GitHubBlog/public/2020/08/13/李宏毅-机器学习21/index.html","38f677dc77dbad807195fdafa56ec7d1"],["E:/GitHubBlog/public/2020/08/14/20200808-0815总结/index.html","f7a3a5a00ca4d1994bb59f65323814d0"],["E:/GitHubBlog/public/2020/09/03/20200816-0903总结/index.html","9fbdacc44802a8d5fe4b4fefa0f88edf"],["E:/GitHubBlog/public/2020/09/16/20200914-0918总结/index.html","e184c6817be680d968deef9749ab9cec"],["E:/GitHubBlog/public/2020/09/25/20200919-0925总结/index.html","46378007803dad9ed850673a543fad78"],["E:/GitHubBlog/public/2020/10/06/《Deep Keyphrase Generation》详细解析！/index.html","044930368742e3f7eeb4494e9295e27b"],["E:/GitHubBlog/public/2020/10/16/20201008-1017总结/index.html","0763141f7336de9482758fcde061371e"],["E:/GitHubBlog/public/2020/10/20/特征选择的几种方法(一)————Filter过滤法/index.html","e68d155c1bb8447e3cc258588415387d"],["E:/GitHubBlog/public/2020/10/21/PCA/index.html","b2ff84cc106bba3fc392bc31319982c9"],["E:/GitHubBlog/public/2020/10/22/20201018-1023总结/index.html","06f1dfaafdae5b095c0377d5e27a7eae"],["E:/GitHubBlog/public/2020/10/24/20201024-1030/index.html","1357614f9e4414ac650b99c19e35cc91"],["E:/GitHubBlog/public/2020/10/30/1030讨论记录/index.html","9a0aac3cc51ba0ecda35d84a5fcd693f"],["E:/GitHubBlog/public/2020/11/03/20201103-总结/index.html","32194ee1de2a4ba29f928c98000d5a0a"],["E:/GitHubBlog/public/2020/11/05/线性回归与逻辑回归/index.html","2a9a6a66e03c1fa13eda3a7e4ed81ed8"],["E:/GitHubBlog/public/2020/11/07/决策树/index.html","2096238adc1434bd137d7bd38b51d265"],["E:/GitHubBlog/public/2020/11/13/20201107-1113总结/index.html","5fa32fd20474f6a06ca2427c16d49388"],["E:/GitHubBlog/public/2020/11/15/20201114-119总结/index.html","3591d3fd79845f56b5b36de859c97d72"],["E:/GitHubBlog/public/2020/11/16/手推一个SVM/index.html","7dbebd92405b4b062ac7e2f07160c5fb"],["E:/GitHubBlog/public/2020/11/27/20201127左右的总结/index.html","f1080ec80c6783528f2edd55db49c588"],["E:/GitHubBlog/public/2020/12/23/第一个深度学习模型（pytorch）/index.html","599df9bacfe77fc2194fccff39e83fcd"],["E:/GitHubBlog/public/2021/01/02/鲍鱼最终版/index.html","5279cac2cc2993920a47528ab965a0c6"],["E:/GitHubBlog/public/2021/01/08/GSA_NumericalExperiment/index.html","42580e3c2124c181685bc18cbc682245"],["E:/GitHubBlog/public/2021/01/16/20210115-01总结/index.html","567c9a135de96f6c88b645fa9468a25f"],["E:/GitHubBlog/public/2021/01/17/20200117-0118总结——OpenNMT阅读（二）/index.html","5f1a567bd12c63804edb0fdfea3f6101"],["E:/GitHubBlog/public/2021/01/17/OpenNMT源码解析/index.html","822a85e73fef89cd37b89109071db60e"],["E:/GitHubBlog/public/2021/01/19/重学数据结构——线性表/index.html","25f923047b4ae76bd1f916e7f722d0fc"],["E:/GitHubBlog/public/2021/01/22/《深度学习入门-基于python的理论与实现》——第二章 感知机/index.html","a640dcdb3620f161cfd482300edf5197"],["E:/GitHubBlog/public/2021/01/26/title复现中的问题/index.html","2afc9a4d6d5a058603c4094f30334ff4"],["E:/GitHubBlog/public/2021/01/26/简单链表题/index.html","db569c13c2d1b65c595b254c75761647"],["E:/GitHubBlog/public/2021/01/28/DeepKeyphraseGeneration/index.html","5bd7741d9fb0727d947bf31eb8b0597c"],["E:/GitHubBlog/public/2021/02/01/title模型详解/index.html","9752f0190d3018db0732a82817d40a83"],["E:/GitHubBlog/public/2021/02/14/中等链表题/index.html","46a9f2068b2e0e005c888678103f9057"],["E:/GitHubBlog/public/2021/02/21/Teacher forcing in RNN/index.html","794ca6d8cb50d4aa8233d2ca4aa215e5"],["E:/GitHubBlog/public/2021/03/11/CopyNet代码消化记录/index.html","156a895fc83d8928e8edc4222487f831"],["E:/GitHubBlog/public/2021/03/11/王道机试指南习题/index.html","d41a579a69d92bf6b3301c3bbe03dfd3"],["E:/GitHubBlog/public/2021/03/13/野生kpg代码挣扎记录/index.html","47f07b11d79fa0cb6d4e4fb8f780f499"],["E:/GitHubBlog/public/archives/2020/01/index.html","3dd9a5f73e514d42033ffb725d673cce"],["E:/GitHubBlog/public/archives/2020/02/index.html","1fe23860a0b92934191e7e4fd3797c33"],["E:/GitHubBlog/public/archives/2020/03/index.html","8a78ffca938c724d79e16f8052e26971"],["E:/GitHubBlog/public/archives/2020/04/index.html","5bc0a7e1e195637048c142d8fa942eff"],["E:/GitHubBlog/public/archives/2020/05/index.html","6a0c4f190de8faae71e37a7398edb5b9"],["E:/GitHubBlog/public/archives/2020/07/index.html","d6d8faff722a335970aa098c97e1806b"],["E:/GitHubBlog/public/archives/2020/08/index.html","fa87bb2e110404f03abac9e5f361be8b"],["E:/GitHubBlog/public/archives/2020/09/index.html","cc72308fea95e972bea04e1e04395f88"],["E:/GitHubBlog/public/archives/2020/10/index.html","902d39ffa14679fa571966577f0798cb"],["E:/GitHubBlog/public/archives/2020/11/index.html","8afd5c4b932920527879d82be1b5c0e6"],["E:/GitHubBlog/public/archives/2020/12/index.html","b682b934e25636c2291dd2715c97eeb0"],["E:/GitHubBlog/public/archives/2020/index.html","fc4c9a508872f2f7c3e649cb41cf6920"],["E:/GitHubBlog/public/archives/2020/page/2/index.html","c41bffcb9392e416e8b3fb6581d5cf5d"],["E:/GitHubBlog/public/archives/2020/page/3/index.html","22363787cb004e892f41443ccba7780b"],["E:/GitHubBlog/public/archives/2020/page/4/index.html","73dd0f58f3d468eb88d0cb5feeb001ed"],["E:/GitHubBlog/public/archives/2020/page/5/index.html","db0bc4a545b12ccfd7758018391fecf3"],["E:/GitHubBlog/public/archives/2021/01/index.html","640edb39fe186d243631e00828103950"],["E:/GitHubBlog/public/archives/2021/02/index.html","a1e09981e5cc0237c7cc02dc719000bd"],["E:/GitHubBlog/public/archives/2021/03/index.html","cb683cb1c03734ad7c51b7815a18d8ca"],["E:/GitHubBlog/public/archives/2021/index.html","08457803512e48cc40acfa70af9d822b"],["E:/GitHubBlog/public/archives/2021/page/2/index.html","f11ac9bfa534906f26414f52203061e6"],["E:/GitHubBlog/public/archives/index.html","5050da51d47c632fd683477da75156f1"],["E:/GitHubBlog/public/archives/page/2/index.html","d7106c5ceaae77d187ddffce17e3df9c"],["E:/GitHubBlog/public/archives/page/3/index.html","8c94c346541ec4827c759658e38b0e2f"],["E:/GitHubBlog/public/archives/page/4/index.html","44d1f71f8e0ce8df7c28d8a64ab4cae4"],["E:/GitHubBlog/public/archives/page/5/index.html","da37a2e6a520afba89c3c465ffc6ca04"],["E:/GitHubBlog/public/archives/page/6/index.html","3058f2b462f978a628662fd4ee313f4c"],["E:/GitHubBlog/public/archives/page/7/index.html","afc3a5c7df6c63073c5804d34772c25e"],["E:/GitHubBlog/public/assets/css/APlayer.min.css","fbe994054426fadb2dff69d824c5c67a"],["E:/GitHubBlog/public/assets/js/APlayer.min.js","8f1017e7a73737e631ff95fa51e4e7d7"],["E:/GitHubBlog/public/assets/js/Meting.min.js","bfac0368480fd344282ec018d28f173d"],["E:/GitHubBlog/public/css/404.css","b1bb50e1fabe41adcec483f6427fb3aa"],["E:/GitHubBlog/public/css/index.css","2389364df340948a5b822422956c2c87"],["E:/GitHubBlog/public/css/var.css","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/img/algolia.svg","fd40b88ac5370a5353a50b8175c1f367"],["E:/GitHubBlog/public/img/avatar.png","6cc4a809d23e3d8946a299ae4ce4e4cd"],["E:/GitHubBlog/public/index.html","541778afffb8f4a3bd03bf0851f021e8"],["E:/GitHubBlog/public/js/copy.js","45aae54bf2e43ac27ecc41eb5e0bacf7"],["E:/GitHubBlog/public/js/fancybox.js","f84d626654b9bbc05720952b3effe062"],["E:/GitHubBlog/public/js/fireworks.js","35933ce61c158ef9c33b5e089fe757ac"],["E:/GitHubBlog/public/js/head.js","347edd99f8e3921b45fa617e839d8182"],["E:/GitHubBlog/public/js/hexo-theme-melody.js","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/js/katex.js","d971ba8b8dee1120ef66744b89cfd2b1"],["E:/GitHubBlog/public/js/scroll.js","a12ad7e37b9e325f8da3d0523d3757c7"],["E:/GitHubBlog/public/js/search/algolia.js","53160985d32d6fd66d98aa0e05b081ac"],["E:/GitHubBlog/public/js/search/local-search.js","1565b508bd866ed6fbd724a3858af5d8"],["E:/GitHubBlog/public/js/sidebar.js","d24db780974e661198eeb2e45f20a28f"],["E:/GitHubBlog/public/js/third-party/anime.min.js","9b4bbe6deb700e1c3606eab732f5eea5"],["E:/GitHubBlog/public/js/third-party/canvas-ribbon.js","09c181544ddff1db701db02ac30453e0"],["E:/GitHubBlog/public/js/third-party/jquery.fancybox.min.js","3c9fa1c1199cd4f874d855ecb1641335"],["E:/GitHubBlog/public/js/third-party/jquery.min.js","c9f5aeeca3ad37bf2aa006139b935f0a"],["E:/GitHubBlog/public/js/third-party/reveal/head.min.js","aad121203010122e05f1766d92385214"],["E:/GitHubBlog/public/js/third-party/reveal/highlight.min.js","44594243bec43813a16371af8fe7e105"],["E:/GitHubBlog/public/js/third-party/reveal/markdown.min.js","7ec4cef5a7fe3f0bf0eb4dc6d7bca114"],["E:/GitHubBlog/public/js/third-party/reveal/marked.min.js","c2a88705e206d71dc21fdc4445349127"],["E:/GitHubBlog/public/js/third-party/reveal/math.min.js","0a278fee2e57c530ab07f7d2d9ea8d96"],["E:/GitHubBlog/public/js/third-party/reveal/notes.min.js","89a0dfae4d706f9c75b317f686c3aa14"],["E:/GitHubBlog/public/js/third-party/reveal/reveal.min.js","8988419d67efb5fe93e291a357e26ec9"],["E:/GitHubBlog/public/js/third-party/reveal/zoom.min.js","9791f96e63e7d534cba2b67d4bda0419"],["E:/GitHubBlog/public/js/third-party/velocity.min.js","64da069aba987ea0512cf610600a56d1"],["E:/GitHubBlog/public/js/third-party/velocity.ui.min.js","c8ca438424a080620f7b2f4ee4b0fff1"],["E:/GitHubBlog/public/js/transition.js","911db4268f0f6621073afcced9e1bfef"],["E:/GitHubBlog/public/js/utils.js","3ff3423d966a1c351e9867813b3f6d36"],["E:/GitHubBlog/public/page/2/index.html","59a352a674410772e64b16ed283d12b7"],["E:/GitHubBlog/public/page/3/index.html","9ed3d5ad86f5b28e50ee604994b683fa"],["E:/GitHubBlog/public/page/4/index.html","0dd28063b6f0bb38969d74be806ddf0f"],["E:/GitHubBlog/public/page/5/index.html","8e1442159f2db4b9144141f7215eec63"],["E:/GitHubBlog/public/page/6/index.html","4882b707fa9c95767ea102a137da6fdd"],["E:/GitHubBlog/public/page/7/index.html","236d378c22db66df137ff11b20517656"],["E:/GitHubBlog/public/tags/Android/index.html","6dfb20aa17adc0abdd2beef0adb3be71"],["E:/GitHubBlog/public/tags/NLP/index.html","25fad5cafa6988b4a1709ae8aca79bd7"],["E:/GitHubBlog/public/tags/NLP/page/2/index.html","083449e622201cb1adf72f0837e05943"],["E:/GitHubBlog/public/tags/NLP/page/3/index.html","7aa10cac806d1769122dae14c9bb485f"],["E:/GitHubBlog/public/tags/R/index.html","52f008bf5316c2a078a86e87934285e7"],["E:/GitHubBlog/public/tags/index.html","e42fb863430b5338de62ef958b3b10c1"],["E:/GitHubBlog/public/tags/java/index.html","f89754bd681ab6266bb6f89b84655f80"],["E:/GitHubBlog/public/tags/java/page/2/index.html","efb2228b6ee38e7f7e4e940db6813d84"],["E:/GitHubBlog/public/tags/leetcode/index.html","73620170ad27c327a83bf63675a32e4a"],["E:/GitHubBlog/public/tags/python/index.html","87ec630634a1b59e25535de7428f03ec"],["E:/GitHubBlog/public/tags/pytorch/index.html","f2ca8ff893c7df086a6f4a2a1c76c846"],["E:/GitHubBlog/public/tags/代码/index.html","085639cb34f133438e7988a4fb6b456d"],["E:/GitHubBlog/public/tags/优化方法/index.html","150b90f0559f1df719f56be346c8223b"],["E:/GitHubBlog/public/tags/总结/index.html","cedd5a1aa5434b9d46ef80b007b858dd"],["E:/GitHubBlog/public/tags/总结/page/2/index.html","21e749b554bde768bf2752274418b061"],["E:/GitHubBlog/public/tags/数据分析/index.html","f3b771d0d60f2bf0a6faa286eae2b595"],["E:/GitHubBlog/public/tags/数据挖掘/index.html","7a6a2ca33b05e88061b630c80776db57"],["E:/GitHubBlog/public/tags/数据结构/index.html","ef5a4fc1f0e30843401bbf3dfb9fa80a"],["E:/GitHubBlog/public/tags/机器学习/index.html","be7380581b04a8bdcc0efe289f501fd1"],["E:/GitHubBlog/public/tags/深度学习/index.html","945f8fbc4909c7321ce0dbadd2bda9af"],["E:/GitHubBlog/public/tags/爬虫/index.html","ea49705d04e39c815225732ca86e75e2"],["E:/GitHubBlog/public/tags/笔记/index.html","9c53a87b57e30093cc9ed1fe877482cf"],["E:/GitHubBlog/public/tags/论文/index.html","81c9757dea8f42c9239df7579796499f"],["E:/GitHubBlog/public/tags/论文/page/2/index.html","5630b1b67171c981c53ae56a41166bb8"],["E:/GitHubBlog/public/tags/论文/page/3/index.html","d5b4ee7d1fbd73c2c7b45036a906c7bd"],["E:/GitHubBlog/public/tags/读书笔记/index.html","831a011f2a9d0678ae24a8cedea077b4"]];
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







