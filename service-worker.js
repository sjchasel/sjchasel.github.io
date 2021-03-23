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

var precacheConfig = [["E:/GitHubBlog/public/2020/01/28/hello-world/index.html","f3adf0ed62ca97197f10e7b1b4f4b9dd"],["E:/GitHubBlog/public/2020/02/08/python爬虫学习总结/index.html","a7bce5049d73e5fbf47a0520ff3f36f9"],["E:/GitHubBlog/public/2020/02/19/数据结构第一周-绪论/index.html","eaeac35708e3df489f484a0d8a271d4b"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-1/index.html","aa98c1788ee9c0679763da8b15aedb51"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-2-创建数据集/index.html","7f36dc2093fcc2ef7623091ff99e074c"],["E:/GitHubBlog/public/2020/03/01/《R语言实战》学习总结-3-图形初阶/index.html","b5db7baa359ff942c205f296143ccadc"],["E:/GitHubBlog/public/2020/03/02/数据结构第二周-线性结构1/index.html","a1d740bdf65ac5e12df7ca7a12c85894"],["E:/GitHubBlog/public/2020/03/03/《R语言实战》学习总结-4-基本数据管理/index.html","e2f6ee5f063e8ce0cc604ded0a4067dc"],["E:/GitHubBlog/public/2020/03/14/数据结构第三周-线性结构2/index.html","0871303ed77aadcfd4d4a4ea1899c09b"],["E:/GitHubBlog/public/2020/03/14/数据结构第四周-栈与队列/index.html","5ad3735935405c74c25cfb1866bb7769"],["E:/GitHubBlog/public/2020/03/17/leetcode记录-1、88、20/index.html","b5a11c9c03f3b18c0b29bc03f1f9db7c"],["E:/GitHubBlog/public/2020/03/20/温度转换程序/index.html","16564977e0803dc71ff4810878ffb731"],["E:/GitHubBlog/public/2020/03/22/数据结构第五周-递归/index.html","125ac6f78438e3c426f67dec23f21c51"],["E:/GitHubBlog/public/2020/03/22/数据结构第六周-串与数组/index.html","f9d7f9f10ca883468d869572d5794bf0"],["E:/GitHubBlog/public/2020/04/02/leetcode记录-52/index.html","d4ed0d905249f67384fc52946478152b"],["E:/GitHubBlog/public/2020/04/17/Android第八周作业-每天第一次打开应用更新数据/index.html","893ed27c8fcdb559ed56fb74d24baefc"],["E:/GitHubBlog/public/2020/05/21/attention模型/index.html","373ad2406e94af1136854a75cf0ead25"],["E:/GitHubBlog/public/2020/05/21/编码器与解码器框架/index.html","37d1e0f696791fda3f06d3af3a9b04b8"],["E:/GitHubBlog/public/2020/07/11/思考，快与慢/index.html","151d69bb7924bcb2b1e26ee53f19f5b1"],["E:/GitHubBlog/public/2020/07/13/20200713-0717总结/index.html","74a8c5ae8bf25a3b081385f75b2c4762"],["E:/GitHubBlog/public/2020/07/13/ZeroShotCeres Zero-Shot Relation Extraction/index.html","93c4d92f7f499093a90d57878fa4b8aa"],["E:/GitHubBlog/public/2020/07/20/Keyphrase Generation A Text Summarization Struggle/index.html","601acb1914d0d9a514b9a538a7ff8e81"],["E:/GitHubBlog/public/2020/07/20/TPR/index.html","921d436804594c3ec5ac26bfb65be3f7"],["E:/GitHubBlog/public/2020/07/23/Keyphrase Generation for Scientific Document Retrieval/index.html","28d35a49f9d25a2f0a47d7feeedb70de"],["E:/GitHubBlog/public/2020/07/24/20200718-0724总结/index.html","d52c73e87333e18c65a797925081f4d5"],["E:/GitHubBlog/public/2020/07/31/20200725-0731总结/index.html","3940c2b2a2d1bd1376e701aafdecbc84"],["E:/GitHubBlog/public/2020/08/07/20200801-0807总结/index.html","ecfd0128867ef0260920a2371c7ef514"],["E:/GitHubBlog/public/2020/08/12/逻辑回归算法/index.html","604d252336ff0882326151141eb9b5bd"],["E:/GitHubBlog/public/2020/08/13/李宏毅-机器学习21/index.html","b55a6b1197570ea7cd7ce023a88ba444"],["E:/GitHubBlog/public/2020/08/14/20200808-0815总结/index.html","2f9103365ba35e3bb6b4cb623465fef8"],["E:/GitHubBlog/public/2020/09/03/20200816-0903总结/index.html","5165e98ff2aa4aad3b47d35b5b4fa619"],["E:/GitHubBlog/public/2020/09/16/20200914-0918总结/index.html","4130a006b30d8aa5309ef599fe238528"],["E:/GitHubBlog/public/2020/09/25/20200919-0925总结/index.html","dbd107c8ca3f9f6084e00e5a0b1ab4a8"],["E:/GitHubBlog/public/2020/10/06/《Deep Keyphrase Generation》详细解析！/index.html","fad5c4eea99764acd7a54557b84ea399"],["E:/GitHubBlog/public/2020/10/16/20201008-1017总结/index.html","08bc52195be452ecbc92f0164ac0d196"],["E:/GitHubBlog/public/2020/10/20/特征选择的几种方法(一)————Filter过滤法/index.html","e04e2769edf57e495db49c6a86fe2656"],["E:/GitHubBlog/public/2020/10/21/PCA/index.html","00ea31b70a1795f349bd154cab84fa0b"],["E:/GitHubBlog/public/2020/10/22/20201018-1023总结/index.html","edcca838ea05a803053931958bf29b8a"],["E:/GitHubBlog/public/2020/10/24/20201024-1030/index.html","8621e75146ef77daf19241a01a4f7f06"],["E:/GitHubBlog/public/2020/10/30/1030讨论记录/index.html","16b3a0c41419b07c038c8026ee88adf8"],["E:/GitHubBlog/public/2020/11/03/20201103-总结/index.html","baa11e7c0a4d3770e4fd3401482c63e6"],["E:/GitHubBlog/public/2020/11/05/线性回归与逻辑回归/index.html","890ac36a03a1d55ed7800c818820660d"],["E:/GitHubBlog/public/2020/11/07/决策树/index.html","2ded6f022d904e7fc0758167def9524d"],["E:/GitHubBlog/public/2020/11/13/20201107-1113总结/index.html","d87220545774dc69ce74a471b1b0deba"],["E:/GitHubBlog/public/2020/11/15/20201114-119总结/index.html","ab22a7376cc82661ee441325801dfafe"],["E:/GitHubBlog/public/2020/11/16/手推一个SVM/index.html","d63eb9111e01877ffd58023fdde23bfa"],["E:/GitHubBlog/public/2020/11/27/20201127左右的总结/index.html","9b5836265e79059054d3a8d34be22489"],["E:/GitHubBlog/public/2020/12/23/第一个深度学习模型（pytorch）/index.html","00c5ce290d824262901d0282fcaa078f"],["E:/GitHubBlog/public/2021/01/02/鲍鱼最终版/index.html","7e955897e1f1e2de8dc979e469db0413"],["E:/GitHubBlog/public/2021/01/08/GSA_NumericalExperiment/index.html","9eb1a09e7569b12ad26aa168701444af"],["E:/GitHubBlog/public/2021/01/16/20210115-01总结/index.html","245c6c81afb60460f28047d107355384"],["E:/GitHubBlog/public/2021/01/17/20200117-0118总结——OpenNMT阅读（二）/index.html","1c478116fa708615fac3c7276c828260"],["E:/GitHubBlog/public/2021/01/17/OpenNMT源码解析/index.html","77acf0af38fc443aab68126fe5a1141a"],["E:/GitHubBlog/public/2021/01/19/重学数据结构——线性表/index.html","786129e84bc83503e9ad682915440bc1"],["E:/GitHubBlog/public/2021/01/22/《深度学习入门-基于python的理论与实现》——第二章 感知机/index.html","27b8f35cd61669097714f0a7143b9c57"],["E:/GitHubBlog/public/2021/01/26/title复现中的问题/index.html","7b1df06d9c5c8695796078480038eda4"],["E:/GitHubBlog/public/2021/01/26/简单链表题/index.html","a4d724ddbfef784968d634a57ef99468"],["E:/GitHubBlog/public/2021/01/28/DeepKeyphraseGeneration/index.html","095dc28a2ae49b54e484c373f48e6624"],["E:/GitHubBlog/public/2021/02/01/title模型详解/index.html","cd06e43e7c917aa498dbc5503eccd063"],["E:/GitHubBlog/public/2021/02/14/中等链表题/index.html","c44e50c0f00459ef27c7fca8318e084a"],["E:/GitHubBlog/public/2021/02/21/Teacher forcing in RNN/index.html","355de02bcc814a5cf3d2dac67d33318e"],["E:/GitHubBlog/public/2021/03/11/CopyNet代码消化记录/index.html","8485e3fe434e0201742d99fe3c382f6f"],["E:/GitHubBlog/public/2021/03/11/王道机试指南习题/index.html","054795a5999d8528fc7ec8e72322299f"],["E:/GitHubBlog/public/2021/03/13/野生kpg代码挣扎记录/index.html","5911dbe5f0effd79d6512ef8b3f987b1"],["E:/GitHubBlog/public/2021/03/19/AcWing代码记录/index.html","5130b9d45249dd8b0b8f3239e7bd89dc"],["E:/GitHubBlog/public/2021/03/20/2021-03-26-week3组会/index.html","f691c9ae1618de77b909b3b7d62fa9db"],["E:/GitHubBlog/public/2021/03/20/20210320-0326总结/index.html","ddc6c7d28e7b6a92629bf2c0d72a1ad4"],["E:/GitHubBlog/public/2021/03/22/NEURAL MACHINE TRANSLATION模型/index.html","38074e1ffb7acc0043114bf2429910cd"],["E:/GitHubBlog/public/2021/03/23/西瓜书学习——第一章 绪论/index.html","e630a70f405b9b869f8c9a91cd273d4b"],["E:/GitHubBlog/public/archives/2020/01/index.html","64dbb7f953d5b9990413a208afcff0b4"],["E:/GitHubBlog/public/archives/2020/02/index.html","25fa4ba86bedb6c8fa44d89a09548e1d"],["E:/GitHubBlog/public/archives/2020/03/index.html","ef5afd9205647cc2ac7b03e4dd0bed36"],["E:/GitHubBlog/public/archives/2020/04/index.html","875d0e49fe4395a2633db18d3a60980f"],["E:/GitHubBlog/public/archives/2020/05/index.html","f60c43ea5f53267501ebe966bd9a6ee2"],["E:/GitHubBlog/public/archives/2020/07/index.html","b6ebdb19a6db6ffe02304e088eedde00"],["E:/GitHubBlog/public/archives/2020/08/index.html","b825b828dbff80c6a510daad6006262a"],["E:/GitHubBlog/public/archives/2020/09/index.html","9ca664f3efb72d2be21aa691b72e0e32"],["E:/GitHubBlog/public/archives/2020/10/index.html","28b557a12d786843fff9f15485b49e5e"],["E:/GitHubBlog/public/archives/2020/11/index.html","4720d57f9220a025f7376982e88283f5"],["E:/GitHubBlog/public/archives/2020/12/index.html","b1ab8804e8c74c8ab2fb00ac075a6d88"],["E:/GitHubBlog/public/archives/2020/index.html","bbd374db930566d72da703fc3521515a"],["E:/GitHubBlog/public/archives/2020/page/2/index.html","b4df574a5f4c1df66ba6f82d5d98dac4"],["E:/GitHubBlog/public/archives/2020/page/3/index.html","95f13261f502b1d490e90351ec792bf1"],["E:/GitHubBlog/public/archives/2020/page/4/index.html","0a78d2ed4f32abe1f3472c9956edf88d"],["E:/GitHubBlog/public/archives/2020/page/5/index.html","d3070285b846207dedfad92cb9912d47"],["E:/GitHubBlog/public/archives/2021/01/index.html","003a86bcaacc4ded3e9358ec10298a80"],["E:/GitHubBlog/public/archives/2021/02/index.html","bce0ae5d280b17f309b532908cd158dc"],["E:/GitHubBlog/public/archives/2021/03/index.html","c24639af34fbdb64e946384f0fb62f7e"],["E:/GitHubBlog/public/archives/2021/index.html","afa5873dd67cbe2c4126b36f57f0bce6"],["E:/GitHubBlog/public/archives/2021/page/2/index.html","7489c72ce66141ef870024fdd30ab165"],["E:/GitHubBlog/public/archives/2021/page/3/index.html","ee55c90e30c044ddd5cdb5443631014a"],["E:/GitHubBlog/public/archives/index.html","18dcd60f7d73f6de9999ad212c71d2db"],["E:/GitHubBlog/public/archives/page/2/index.html","a5ed45f82b343de5e5eb4f2e04935529"],["E:/GitHubBlog/public/archives/page/3/index.html","23e3d15c8adbf79b3ce1780d94aea867"],["E:/GitHubBlog/public/archives/page/4/index.html","bbaaf1e4dacc3f9b4668492c408e10d3"],["E:/GitHubBlog/public/archives/page/5/index.html","5c2fc1319f8a295a9138b197e2d9b120"],["E:/GitHubBlog/public/archives/page/6/index.html","4fd9b4cb277485091289f79bbe8e97b3"],["E:/GitHubBlog/public/archives/page/7/index.html","9dc0d024d26b92c26e0e5c6f585c57fa"],["E:/GitHubBlog/public/assets/css/APlayer.min.css","fbe994054426fadb2dff69d824c5c67a"],["E:/GitHubBlog/public/assets/js/APlayer.min.js","8f1017e7a73737e631ff95fa51e4e7d7"],["E:/GitHubBlog/public/assets/js/Meting.min.js","bfac0368480fd344282ec018d28f173d"],["E:/GitHubBlog/public/css/404.css","b1bb50e1fabe41adcec483f6427fb3aa"],["E:/GitHubBlog/public/css/index.css","2389364df340948a5b822422956c2c87"],["E:/GitHubBlog/public/css/var.css","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/img/algolia.svg","fd40b88ac5370a5353a50b8175c1f367"],["E:/GitHubBlog/public/img/avatar.png","6cc4a809d23e3d8946a299ae4ce4e4cd"],["E:/GitHubBlog/public/index.html","2c1b7edec1059aa2d2345696c617d426"],["E:/GitHubBlog/public/js/copy.js","45aae54bf2e43ac27ecc41eb5e0bacf7"],["E:/GitHubBlog/public/js/fancybox.js","f84d626654b9bbc05720952b3effe062"],["E:/GitHubBlog/public/js/fireworks.js","35933ce61c158ef9c33b5e089fe757ac"],["E:/GitHubBlog/public/js/head.js","347edd99f8e3921b45fa617e839d8182"],["E:/GitHubBlog/public/js/hexo-theme-melody.js","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/js/katex.js","d971ba8b8dee1120ef66744b89cfd2b1"],["E:/GitHubBlog/public/js/scroll.js","a12ad7e37b9e325f8da3d0523d3757c7"],["E:/GitHubBlog/public/js/search/algolia.js","53160985d32d6fd66d98aa0e05b081ac"],["E:/GitHubBlog/public/js/search/local-search.js","1565b508bd866ed6fbd724a3858af5d8"],["E:/GitHubBlog/public/js/sidebar.js","d24db780974e661198eeb2e45f20a28f"],["E:/GitHubBlog/public/js/third-party/anime.min.js","9b4bbe6deb700e1c3606eab732f5eea5"],["E:/GitHubBlog/public/js/third-party/canvas-ribbon.js","09c181544ddff1db701db02ac30453e0"],["E:/GitHubBlog/public/js/third-party/jquery.fancybox.min.js","3c9fa1c1199cd4f874d855ecb1641335"],["E:/GitHubBlog/public/js/third-party/jquery.min.js","c9f5aeeca3ad37bf2aa006139b935f0a"],["E:/GitHubBlog/public/js/third-party/reveal/head.min.js","aad121203010122e05f1766d92385214"],["E:/GitHubBlog/public/js/third-party/reveal/highlight.min.js","44594243bec43813a16371af8fe7e105"],["E:/GitHubBlog/public/js/third-party/reveal/markdown.min.js","7ec4cef5a7fe3f0bf0eb4dc6d7bca114"],["E:/GitHubBlog/public/js/third-party/reveal/marked.min.js","c2a88705e206d71dc21fdc4445349127"],["E:/GitHubBlog/public/js/third-party/reveal/math.min.js","0a278fee2e57c530ab07f7d2d9ea8d96"],["E:/GitHubBlog/public/js/third-party/reveal/notes.min.js","89a0dfae4d706f9c75b317f686c3aa14"],["E:/GitHubBlog/public/js/third-party/reveal/reveal.min.js","8988419d67efb5fe93e291a357e26ec9"],["E:/GitHubBlog/public/js/third-party/reveal/zoom.min.js","9791f96e63e7d534cba2b67d4bda0419"],["E:/GitHubBlog/public/js/third-party/velocity.min.js","64da069aba987ea0512cf610600a56d1"],["E:/GitHubBlog/public/js/third-party/velocity.ui.min.js","c8ca438424a080620f7b2f4ee4b0fff1"],["E:/GitHubBlog/public/js/transition.js","911db4268f0f6621073afcced9e1bfef"],["E:/GitHubBlog/public/js/utils.js","3ff3423d966a1c351e9867813b3f6d36"],["E:/GitHubBlog/public/page/2/index.html","6aa7b6c7818ccb73d6851dd2f4c25e8b"],["E:/GitHubBlog/public/page/3/index.html","71ebece35852ef7637906837316fac1d"],["E:/GitHubBlog/public/page/4/index.html","79891df28b82dbe0ec23a04e04e9d513"],["E:/GitHubBlog/public/page/5/index.html","8ccd7302cb906dcf2fb3955eb4a48841"],["E:/GitHubBlog/public/page/6/index.html","3110d12b8c23e6a7dd4a13e2a10ae299"],["E:/GitHubBlog/public/page/7/index.html","4c9f7b880d30a029d8551664364572db"],["E:/GitHubBlog/public/tags/Android/index.html","7638b7726abc4a45d0e8ef57dd068d87"],["E:/GitHubBlog/public/tags/NLP/index.html","3c1c5725bf15e1078483999f9d1ed8b6"],["E:/GitHubBlog/public/tags/NLP/page/2/index.html","2e46eb071cbc45ca531010b730fed86c"],["E:/GitHubBlog/public/tags/NLP/page/3/index.html","47b93c2c0eeca60d329bf5bd0708a749"],["E:/GitHubBlog/public/tags/R/index.html","2ac434bad10ae72a5d14742cdca7b150"],["E:/GitHubBlog/public/tags/index.html","71a47e8c02555fde7ac295f34ef7e622"],["E:/GitHubBlog/public/tags/java/index.html","e4771ddf9d15ca439b70a0443ca71650"],["E:/GitHubBlog/public/tags/java/page/2/index.html","6eacca823f0b8479e2c721e4cbe25c4d"],["E:/GitHubBlog/public/tags/leetcode/index.html","ede56bceea8c84d9662f94d25ea9b411"],["E:/GitHubBlog/public/tags/python/index.html","1aa8d15c7c32dd30b8af42cecefe1ac0"],["E:/GitHubBlog/public/tags/pytorch/index.html","32695a76cae1373d6e913eca21b7e64c"],["E:/GitHubBlog/public/tags/代码/index.html","d8e25c4236e3bf0669143904d90ef0e1"],["E:/GitHubBlog/public/tags/优化方法/index.html","2ccb3dc90d9b38a3e566104a5cb993cb"],["E:/GitHubBlog/public/tags/总结/index.html","e9194ca4d55d3b2d0f579d4e0417e45e"],["E:/GitHubBlog/public/tags/总结/page/2/index.html","7fe5998b0874ff405489c6ca8a445a88"],["E:/GitHubBlog/public/tags/数据分析/index.html","b6224d18c18c9a1eb86bb6752ee5217f"],["E:/GitHubBlog/public/tags/数据挖掘/index.html","30690a9ec3ca74c100e6fc4c1ae27b2d"],["E:/GitHubBlog/public/tags/数据结构/index.html","0db4b77d28b39539af784c1c800b3584"],["E:/GitHubBlog/public/tags/机器学习/index.html","cc3edb0c08f33836f9c5e76768fef0d5"],["E:/GitHubBlog/public/tags/深度学习/index.html","9b2d1bc3fecbac9376ef362d1a087468"],["E:/GitHubBlog/public/tags/爬虫/index.html","ac8c9fd46621a391db75f4b6b821f63c"],["E:/GitHubBlog/public/tags/笔记/index.html","f233029f35d36e0e9a47b83ffb268e50"],["E:/GitHubBlog/public/tags/论文/index.html","ff0e7aa205cc68b49000f84fa8ca124a"],["E:/GitHubBlog/public/tags/论文/page/2/index.html","19360ffe41f0b89d1ec6195998154246"],["E:/GitHubBlog/public/tags/论文/page/3/index.html","9b081526e3013c25201db2bd25f78b6f"],["E:/GitHubBlog/public/tags/读书笔记/index.html","d8e8ef394491eaad10e440ccea7bfaeb"]];
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







