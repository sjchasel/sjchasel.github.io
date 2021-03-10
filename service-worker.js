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

var precacheConfig = [["E:/GitHubBlog/public/2020/01/28/hello-world/index.html","e3dc7f9929dc6fe2168ad329fa94015a"],["E:/GitHubBlog/public/2020/02/08/python爬虫学习总结/index.html","d0cd0160a24dc334e2b93157ff43c26d"],["E:/GitHubBlog/public/2020/02/19/数据结构第一周-绪论/index.html","f807fa4b037b3461401353a277f4ed68"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-1/index.html","dd3696bce9d7e8a79d350641333ce834"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-2-创建数据集/index.html","5611594ba996eef9267e2f1cd3988df0"],["E:/GitHubBlog/public/2020/03/01/《R语言实战》学习总结-3-图形初阶/index.html","34ba8f78fbf8ef27dc2b4ff1dcbf7d21"],["E:/GitHubBlog/public/2020/03/02/数据结构第二周-线性结构1/index.html","c3f2cc30543905e0a89c4040ca71efed"],["E:/GitHubBlog/public/2020/03/03/《R语言实战》学习总结-4-基本数据管理/index.html","ebb8847f0634353f30073632831866b1"],["E:/GitHubBlog/public/2020/03/14/数据结构第三周-线性结构2/index.html","1d90513f839257b89d115f883cabf66f"],["E:/GitHubBlog/public/2020/03/14/数据结构第四周-栈与队列/index.html","7315e1bc30e2938c59a8c79956cbbceb"],["E:/GitHubBlog/public/2020/03/17/leetcode记录-1、88、20/index.html","c8b119df6e6e3ac2c6c66f0f11db001e"],["E:/GitHubBlog/public/2020/03/20/温度转换程序/index.html","5beec82648f8e3c2d420dc0c5db12174"],["E:/GitHubBlog/public/2020/03/22/数据结构第五周-递归/index.html","559a97aebf11a6ed3a9467982c37d8fa"],["E:/GitHubBlog/public/2020/03/22/数据结构第六周-串与数组/index.html","f2a1a055095ef25bf2a9522704d80d12"],["E:/GitHubBlog/public/2020/04/02/leetcode记录-52/index.html","de1d866f9ad32d77cfbe6bdcd935c4f5"],["E:/GitHubBlog/public/2020/04/17/Android第八周作业-每天第一次打开应用更新数据/index.html","acbd4208a6dc7dbf224e013581f2e18f"],["E:/GitHubBlog/public/2020/05/21/attention模型/index.html","190c41642909650dfe6555be5aebf529"],["E:/GitHubBlog/public/2020/05/21/编码器与解码器框架/index.html","8f05213b60adf58d38aef159a4b40d4e"],["E:/GitHubBlog/public/2020/07/11/思考，快与慢/index.html","31fc0133aa2932f715424ba74a21ac60"],["E:/GitHubBlog/public/2020/07/13/20200713-0717总结/index.html","62eb107d63dcab36eee93e4bb938739a"],["E:/GitHubBlog/public/2020/07/13/ZeroShotCeres Zero-Shot Relation Extraction/index.html","7213a933d4a5a084594d548ecff0910d"],["E:/GitHubBlog/public/2020/07/20/Keyphrase Generation A Text Summarization Struggle/index.html","540ed3924ea07c8caa420f9774c734b0"],["E:/GitHubBlog/public/2020/07/20/TPR/index.html","ebf6c3a9551af7c24e20a0dd5515f810"],["E:/GitHubBlog/public/2020/07/23/Keyphrase Generation for Scientific Document Retrieval/index.html","ffd678fb5fc8fa0a43390911b9e9d218"],["E:/GitHubBlog/public/2020/07/24/20200718-0724总结/index.html","1386fd731dcea613e29a724ecf48fd22"],["E:/GitHubBlog/public/2020/07/31/20200725-0731总结/index.html","995d7361a24aa5d57afcdd7777e2bfe7"],["E:/GitHubBlog/public/2020/08/07/20200801-0807总结/index.html","acbab7178c8c4caede990370f088a820"],["E:/GitHubBlog/public/2020/08/12/逻辑回归算法/index.html","61b9358e5dd847003f5071f26b683ee0"],["E:/GitHubBlog/public/2020/08/13/李宏毅-机器学习21/index.html","4d7d21fd93688306c2588469d0a8fb49"],["E:/GitHubBlog/public/2020/08/14/20200808-0815总结/index.html","3c339d610d0e34de95492e9803686721"],["E:/GitHubBlog/public/2020/09/03/20200816-0903总结/index.html","b0bbbbe7e9d77c3c50a4142a88d536f0"],["E:/GitHubBlog/public/2020/09/16/20200914-0918总结/index.html","69d613bd55ed475dcd04b340cda59802"],["E:/GitHubBlog/public/2020/09/25/20200919-0925总结/index.html","2da2669ce612927267ba9c91f9964851"],["E:/GitHubBlog/public/2020/10/06/《Deep Keyphrase Generation》详细解析！/index.html","c1af04d3ad3b432b89e1f52bb40cdb88"],["E:/GitHubBlog/public/2020/10/16/20201008-1017总结/index.html","791d3bed409014e53fcbdca2fac0939b"],["E:/GitHubBlog/public/2020/10/20/特征选择的几种方法(一)————Filter过滤法/index.html","3b8c2524c703f39f9af722348cafdc81"],["E:/GitHubBlog/public/2020/10/21/PCA/index.html","63c6660fca96b1b0e9c3648c552c5202"],["E:/GitHubBlog/public/2020/10/22/20201018-1023总结/index.html","a36d2751e4e013bcd8e8d674c08f1a99"],["E:/GitHubBlog/public/2020/10/24/20201024-1030/index.html","caee6f788fd94d95d7ccacb5518128c0"],["E:/GitHubBlog/public/2020/10/30/1030讨论记录/index.html","90390d042e218b29adf9a4e23b3cb73f"],["E:/GitHubBlog/public/2020/11/03/20201103-总结/index.html","15f78b329e5640941a8c4768eacc1a39"],["E:/GitHubBlog/public/2020/11/05/线性回归与逻辑回归/index.html","17b83b2ffe4320df0e3620f81c7d32ff"],["E:/GitHubBlog/public/2020/11/07/决策树/index.html","03537de16c9a98158ddebf1f3dc0c90d"],["E:/GitHubBlog/public/2020/11/13/20201107-1113总结/index.html","bc228f3ca9e3a3e3ab387cec3536385a"],["E:/GitHubBlog/public/2020/11/15/20201114-119总结/index.html","f77cb14e6e3a37182a933d41ebce1796"],["E:/GitHubBlog/public/2020/11/16/手推一个SVM/index.html","346a4d51876b43c41d6e8bd8fd72d16b"],["E:/GitHubBlog/public/2020/11/27/20201127左右的总结/index.html","f7d494388977d5909af81c323d6d2dc4"],["E:/GitHubBlog/public/2020/12/23/第一个深度学习模型（pytorch）/index.html","828a45f45c34c805921100c4290ec3d8"],["E:/GitHubBlog/public/2021/01/02/鲍鱼最终版/index.html","4fc2b9792edde54203db815ce3207f68"],["E:/GitHubBlog/public/2021/01/08/GSA_NumericalExperiment/index.html","6ab9853bf2cf203a5cddf35f5a6721b4"],["E:/GitHubBlog/public/2021/01/16/20210115-01总结/index.html","bcac842fec7d9df6f1a4e16a8d8b6546"],["E:/GitHubBlog/public/2021/01/17/20200117-0118总结——OpenNMT阅读（二）/index.html","4d8b4b2a9f2771a10dd1e18f687332d6"],["E:/GitHubBlog/public/2021/01/17/OpenNMT源码解析/index.html","0bdf247f2be78ee8a5f4b9ba1c41902c"],["E:/GitHubBlog/public/2021/01/19/重学数据结构——线性表/index.html","54cf400598ce0c4e8298ed03557876d5"],["E:/GitHubBlog/public/2021/01/22/《深度学习入门-基于python的理论与实现》——第二章 感知机/index.html","0159d56b4e311fceba3510072ef67b2f"],["E:/GitHubBlog/public/2021/01/22/代码理论/index.html","b45af0c3fe793419fbef1cf1f4af5730"],["E:/GitHubBlog/public/2021/01/26/title复现中的问题/index.html","a6869f10c9b9ac2d7f504e5aa4a30df0"],["E:/GitHubBlog/public/2021/01/26/简单链表题/index.html","4de790cfe835d41c02bb96dd404998f3"],["E:/GitHubBlog/public/2021/01/28/DeepKeyphraseGeneration/index.html","84964b34a0fed61ef4b68bff6d781daf"],["E:/GitHubBlog/public/2021/02/01/title模型详解/index.html","ed8fefcd7c1049bfff30b6fa0985c431"],["E:/GitHubBlog/public/2021/02/14/中等链表题/index.html","d855503f219b242bf0bac70c4d0ea13a"],["E:/GitHubBlog/public/2021/02/21/Teacher forcing in RNN/index.html","9fafa9b0b5988a8e3763cf0d90b338d8"],["E:/GitHubBlog/public/archives/2020/01/index.html","141bff26807a479f76502cd91d2a40dc"],["E:/GitHubBlog/public/archives/2020/02/index.html","0e7b7966686942e3cf4bfbcabaca0256"],["E:/GitHubBlog/public/archives/2020/03/index.html","b58067353c12dd4afa42dee4fa1dcc76"],["E:/GitHubBlog/public/archives/2020/04/index.html","2d259941297b5d1e1dd908ae1b598649"],["E:/GitHubBlog/public/archives/2020/05/index.html","bbe2f3f0b4bb49b5edab0eeda0174b96"],["E:/GitHubBlog/public/archives/2020/07/index.html","188b74f55dc6ebaf3dd0fa4689e0a6eb"],["E:/GitHubBlog/public/archives/2020/08/index.html","f0ce74999033882e03271b051a2b081f"],["E:/GitHubBlog/public/archives/2020/09/index.html","561ca45571d2832f7476805f8a9b2f4a"],["E:/GitHubBlog/public/archives/2020/10/index.html","43f60fc6bbe3cef2423cb75753e8bf85"],["E:/GitHubBlog/public/archives/2020/11/index.html","32d6d5c210b0dabc00687109ee267cf6"],["E:/GitHubBlog/public/archives/2020/12/index.html","6a70a142ee4639797995ca94b2c0f3e5"],["E:/GitHubBlog/public/archives/2020/index.html","abfa1ede6ed33fa8284b508c88ca29e1"],["E:/GitHubBlog/public/archives/2020/page/2/index.html","0650b94f5842c2c36461cb91a14f34f6"],["E:/GitHubBlog/public/archives/2020/page/3/index.html","72b5db8f0c91ca31863be7191d7cc652"],["E:/GitHubBlog/public/archives/2020/page/4/index.html","31191ca11f5a3cd84dead05eb87654d2"],["E:/GitHubBlog/public/archives/2020/page/5/index.html","a38dae24ba3edee8cd85fa15189e79c1"],["E:/GitHubBlog/public/archives/2021/01/index.html","1f2b3c992edb1c3779986566a59f7929"],["E:/GitHubBlog/public/archives/2021/01/page/2/index.html","e50d39ea92b1a8246153344e8cd27fdc"],["E:/GitHubBlog/public/archives/2021/02/index.html","4081bdf743c7ca973d51bee0b90eb064"],["E:/GitHubBlog/public/archives/2021/index.html","019121e2e6420d55197cb1e04be57dd3"],["E:/GitHubBlog/public/archives/2021/page/2/index.html","ab1da1302b6ad453cb91c4319e517caf"],["E:/GitHubBlog/public/archives/index.html","98f74b7ce0e4d0cdb4e3314171a07cf7"],["E:/GitHubBlog/public/archives/page/2/index.html","0dff34b082aaf65cf210592dba98357d"],["E:/GitHubBlog/public/archives/page/3/index.html","9c66e98f5e474a38ff5723343953859a"],["E:/GitHubBlog/public/archives/page/4/index.html","d0833f6ef074574f64a2a9890a5e46b6"],["E:/GitHubBlog/public/archives/page/5/index.html","1b01c2919717cdb8e3b11529b861fb50"],["E:/GitHubBlog/public/archives/page/6/index.html","ef2ac6a4b0b338eb3a966e3ceaf07804"],["E:/GitHubBlog/public/archives/page/7/index.html","e99dcd5ede2b1b837b0e8ed50038f9c1"],["E:/GitHubBlog/public/assets/css/APlayer.min.css","fbe994054426fadb2dff69d824c5c67a"],["E:/GitHubBlog/public/assets/js/APlayer.min.js","8f1017e7a73737e631ff95fa51e4e7d7"],["E:/GitHubBlog/public/assets/js/Meting.min.js","bfac0368480fd344282ec018d28f173d"],["E:/GitHubBlog/public/css/404.css","b1bb50e1fabe41adcec483f6427fb3aa"],["E:/GitHubBlog/public/css/index.css","2389364df340948a5b822422956c2c87"],["E:/GitHubBlog/public/css/var.css","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/img/algolia.svg","fd40b88ac5370a5353a50b8175c1f367"],["E:/GitHubBlog/public/img/avatar.png","6cc4a809d23e3d8946a299ae4ce4e4cd"],["E:/GitHubBlog/public/index.html","0bd389e8fb63308f1b2eca8e4a05a48a"],["E:/GitHubBlog/public/js/copy.js","45aae54bf2e43ac27ecc41eb5e0bacf7"],["E:/GitHubBlog/public/js/fancybox.js","f84d626654b9bbc05720952b3effe062"],["E:/GitHubBlog/public/js/fireworks.js","35933ce61c158ef9c33b5e089fe757ac"],["E:/GitHubBlog/public/js/head.js","347edd99f8e3921b45fa617e839d8182"],["E:/GitHubBlog/public/js/hexo-theme-melody.js","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/js/katex.js","d971ba8b8dee1120ef66744b89cfd2b1"],["E:/GitHubBlog/public/js/scroll.js","a12ad7e37b9e325f8da3d0523d3757c7"],["E:/GitHubBlog/public/js/search/algolia.js","53160985d32d6fd66d98aa0e05b081ac"],["E:/GitHubBlog/public/js/search/local-search.js","1565b508bd866ed6fbd724a3858af5d8"],["E:/GitHubBlog/public/js/sidebar.js","d24db780974e661198eeb2e45f20a28f"],["E:/GitHubBlog/public/js/third-party/anime.min.js","9b4bbe6deb700e1c3606eab732f5eea5"],["E:/GitHubBlog/public/js/third-party/canvas-ribbon.js","09c181544ddff1db701db02ac30453e0"],["E:/GitHubBlog/public/js/third-party/jquery.fancybox.min.js","3c9fa1c1199cd4f874d855ecb1641335"],["E:/GitHubBlog/public/js/third-party/jquery.min.js","c9f5aeeca3ad37bf2aa006139b935f0a"],["E:/GitHubBlog/public/js/third-party/reveal/head.min.js","aad121203010122e05f1766d92385214"],["E:/GitHubBlog/public/js/third-party/reveal/highlight.min.js","44594243bec43813a16371af8fe7e105"],["E:/GitHubBlog/public/js/third-party/reveal/markdown.min.js","7ec4cef5a7fe3f0bf0eb4dc6d7bca114"],["E:/GitHubBlog/public/js/third-party/reveal/marked.min.js","c2a88705e206d71dc21fdc4445349127"],["E:/GitHubBlog/public/js/third-party/reveal/math.min.js","0a278fee2e57c530ab07f7d2d9ea8d96"],["E:/GitHubBlog/public/js/third-party/reveal/notes.min.js","89a0dfae4d706f9c75b317f686c3aa14"],["E:/GitHubBlog/public/js/third-party/reveal/reveal.min.js","8988419d67efb5fe93e291a357e26ec9"],["E:/GitHubBlog/public/js/third-party/reveal/zoom.min.js","9791f96e63e7d534cba2b67d4bda0419"],["E:/GitHubBlog/public/js/third-party/velocity.min.js","64da069aba987ea0512cf610600a56d1"],["E:/GitHubBlog/public/js/third-party/velocity.ui.min.js","c8ca438424a080620f7b2f4ee4b0fff1"],["E:/GitHubBlog/public/js/transition.js","911db4268f0f6621073afcced9e1bfef"],["E:/GitHubBlog/public/js/utils.js","3ff3423d966a1c351e9867813b3f6d36"],["E:/GitHubBlog/public/page/2/index.html","b42b1c54db07589a83680bae4fa4f84f"],["E:/GitHubBlog/public/page/3/index.html","039aebd71698c896d9d05344280ae924"],["E:/GitHubBlog/public/page/4/index.html","fe3c67c72da8f1fa2d20b632dbb0c8c9"],["E:/GitHubBlog/public/page/5/index.html","976cdcbaad8010cc9984ee4887ed0335"],["E:/GitHubBlog/public/page/6/index.html","2fa7e5505afb049f6e1b8c4f2f2d51df"],["E:/GitHubBlog/public/page/7/index.html","3eeef10a1872094885a3f8b49dad4d43"],["E:/GitHubBlog/public/tags/Android/index.html","d259b0eec20a37cb96fca69e6f26a1da"],["E:/GitHubBlog/public/tags/NLP/index.html","dc8b3c8e3edd2c8f8344547ba58c24b2"],["E:/GitHubBlog/public/tags/NLP/page/2/index.html","f71c29729bcef75401ed935906ecbd49"],["E:/GitHubBlog/public/tags/NLP/page/3/index.html","2a6bf3bee3a1ca61ad2a1eb8d60b8d4d"],["E:/GitHubBlog/public/tags/R/index.html","044e5a70632a06aa6dfbbd3accbe3972"],["E:/GitHubBlog/public/tags/index.html","2c7e9b08814f7dce0a18d2f7293d990c"],["E:/GitHubBlog/public/tags/java/index.html","dfec423d69e2e5c0d2eeba62a7d25ea0"],["E:/GitHubBlog/public/tags/java/page/2/index.html","a07c2bd3faab47c0658938c128aef1d8"],["E:/GitHubBlog/public/tags/leetcode/index.html","98fd2d036b969a77efbc946dba17d1d3"],["E:/GitHubBlog/public/tags/python/index.html","b3fbaf84d39d426d49be844ff871419a"],["E:/GitHubBlog/public/tags/pytorch/index.html","84fade5e439c191d43fc4fc5a04a91c0"],["E:/GitHubBlog/public/tags/优化方法/index.html","e7f01bb7a1b4357e2b48a5ce2662dfd0"],["E:/GitHubBlog/public/tags/总结/index.html","1f84fcdb15f19d612be899dc603c3169"],["E:/GitHubBlog/public/tags/总结/page/2/index.html","0c928773c5f425d2af81cdd84ba45eeb"],["E:/GitHubBlog/public/tags/数据分析/index.html","74da39b4dd1241bd3772945608183e28"],["E:/GitHubBlog/public/tags/数据挖掘/index.html","06ec7935a90730b5c9d99b16fc742bea"],["E:/GitHubBlog/public/tags/数据结构/index.html","140d22efbaaf21201b61f31a0bc30d41"],["E:/GitHubBlog/public/tags/机器学习/index.html","564f47cfe0c073fc910d08aa4d589d84"],["E:/GitHubBlog/public/tags/深度学习/index.html","828193d0bf51c969c4d8cd02a34b450f"],["E:/GitHubBlog/public/tags/爬虫/index.html","2fe70fa5fdbfa9b5b57ac4597fe269c0"],["E:/GitHubBlog/public/tags/笔记/index.html","f87915b43362454e9f9ebe3d6d3cb85c"],["E:/GitHubBlog/public/tags/论文/index.html","3a819faf7f06286c27b6440464b273dd"],["E:/GitHubBlog/public/tags/论文/page/2/index.html","1d0073004117d823f580af14ce9ef427"],["E:/GitHubBlog/public/tags/论文/page/3/index.html","0f63f88719d0044c23e0959749b76874"],["E:/GitHubBlog/public/tags/读书笔记/index.html","ea9b2d25833deea95a12cfa53ace75d6"]];
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







