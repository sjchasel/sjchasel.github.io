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

var precacheConfig = [["E:/GitHubBlog/public/2020/01/28/hello-world/index.html","2ea04bb1d96aa66e6a395e533517ee02"],["E:/GitHubBlog/public/2020/02/08/python爬虫学习总结/index.html","bff2074e2d3add91cf36604f9ba6cfe5"],["E:/GitHubBlog/public/2020/02/19/数据结构第一周-绪论/index.html","23515dbf6bcc8a381b3eab9c00ed32d9"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-1/index.html","bdc86631b30335bc3cecd7ccfe5625fb"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-2-创建数据集/index.html","45e2052300eb531c7baae34731125e13"],["E:/GitHubBlog/public/2020/03/01/《R语言实战》学习总结-3-图形初阶/index.html","92bb8649eb2804938373b8f733a2cb1e"],["E:/GitHubBlog/public/2020/03/02/数据结构第二周-线性结构1/index.html","9c20b850057d613580206d68de6aa4e9"],["E:/GitHubBlog/public/2020/03/03/《R语言实战》学习总结-4-基本数据管理/index.html","8fabe7b2912c147a189855120fb8bf13"],["E:/GitHubBlog/public/2020/03/14/数据结构第三周-线性结构2/index.html","810b0b8f8136a3d57a39b4deb757f950"],["E:/GitHubBlog/public/2020/03/14/数据结构第四周-栈与队列/index.html","6059fb9c81bf9181d945271ad16919b4"],["E:/GitHubBlog/public/2020/03/17/leetcode记录-1、88、20/index.html","cd28e32a4e1c07b02aa9069dab722c64"],["E:/GitHubBlog/public/2020/03/20/温度转换程序/index.html","bb969b567f50bcc54d6c5bdeeb20a23c"],["E:/GitHubBlog/public/2020/03/22/数据结构第五周-递归/index.html","3e5fa6ee859d3c19d538429a6c9d5abe"],["E:/GitHubBlog/public/2020/03/22/数据结构第六周-串与数组/index.html","1a95be934d7c6a6bfca1baa1a40484dc"],["E:/GitHubBlog/public/2020/04/02/leetcode记录-52/index.html","8067b17450708c4a0d0bc9d55a18a5de"],["E:/GitHubBlog/public/2020/04/17/Android第八周作业-每天第一次打开应用更新数据/index.html","885459ef399dc48efd317bd5a79f95a0"],["E:/GitHubBlog/public/2020/05/21/attention模型/index.html","8b9ec68a0eec7988e2d9f31b0f1f6ff3"],["E:/GitHubBlog/public/2020/05/21/编码器与解码器框架/index.html","228d590a7b5ee53bd2a7181111f6bdd5"],["E:/GitHubBlog/public/2020/07/11/思考，快与慢/index.html","ef21c2bbd6288ee5d86e1f2ed63ffd8f"],["E:/GitHubBlog/public/2020/07/13/20200713-0717总结/index.html","08b0e3be83b3cc3801a34db548e6111a"],["E:/GitHubBlog/public/2020/07/13/ZeroShotCeres Zero-Shot Relation Extraction/index.html","1b91fbe03f00c44daa548f78b16c5893"],["E:/GitHubBlog/public/2020/07/20/Keyphrase Generation A Text Summarization Struggle/index.html","21e77781dce1196254148fc3dc900634"],["E:/GitHubBlog/public/2020/07/20/TPR/index.html","c523b660ced02b3b166ca1cfa1fdb3b9"],["E:/GitHubBlog/public/2020/07/23/Keyphrase Generation for Scientific Document Retrieval/index.html","a0c0e08efcfa315cda50fa0f0c660c1d"],["E:/GitHubBlog/public/2020/07/24/20200718-0724总结/index.html","2f1f8213639f09b5dc6d84ffebc19798"],["E:/GitHubBlog/public/2020/07/31/20200725-0731总结/index.html","2e83b349c96f820498c4ac0904677446"],["E:/GitHubBlog/public/2020/08/07/20200801-0807总结/index.html","686508f1498c821551e6962a893a68f2"],["E:/GitHubBlog/public/2020/08/13/李宏毅-机器学习21/index.html","bc56176da06df6ae1c7eddd37ed35415"],["E:/GitHubBlog/public/2020/08/14/20200808-0815总结/index.html","d9fb1385857b04ac8079b6bfe250e4f5"],["E:/GitHubBlog/public/2020/09/03/20200816-0903总结/index.html","ee833d639f48219583f4fa75f6af3e3a"],["E:/GitHubBlog/public/2020/09/16/20200914-0918总结/index.html","cf66d4fb0780e4c45fafaf7c983163f9"],["E:/GitHubBlog/public/2020/09/25/20200919-0925总结/index.html","f0a5cc76ab76f797ed42b799974e4356"],["E:/GitHubBlog/public/2020/10/06/《Deep Keyphrase Generation》详细解析！/index.html","90e28f852f7269d85f3a521bf66d3162"],["E:/GitHubBlog/public/2020/10/16/20201008-1017总结/index.html","cb92066c3bc773632484c777724a8ff1"],["E:/GitHubBlog/public/2020/10/20/特征选择的几种方法(一)————Filter过滤法/index.html","1c1ae8af11c98ae3086d1f28177eed0b"],["E:/GitHubBlog/public/2020/10/21/PCA/index.html","2f4084bd8a262e342c3a6ebe53463a04"],["E:/GitHubBlog/public/2020/10/22/20201018-1023总结/index.html","51809369aeefccb87089c346184d2317"],["E:/GitHubBlog/public/2020/10/24/20201024-1030/index.html","5d05c186eec3d0d1474691a6c5972e26"],["E:/GitHubBlog/public/2020/10/30/1030讨论记录/index.html","dc8f9a611a311840be7d342d3c1cd92a"],["E:/GitHubBlog/public/2020/11/03/20201103-总结/index.html","722e2644f28edb634ac2df99498e0044"],["E:/GitHubBlog/public/2020/11/05/线性回归与逻辑回归/index.html","26a0b26d3187309c32bdbb2877a81262"],["E:/GitHubBlog/public/2020/11/07/决策树/index.html","3e4b04aa6851263d07e5dddaddef3be6"],["E:/GitHubBlog/public/2020/11/13/20201107-1113总结/index.html","849ae2f53f86f8760fea24054ea67609"],["E:/GitHubBlog/public/2020/11/15/20201114-119总结/index.html","92546ef7ca80fb49afd2f338f093ba84"],["E:/GitHubBlog/public/2020/11/16/手推一个SVM/index.html","fcde8b5d652d8a32c11cbf1fbc492bac"],["E:/GitHubBlog/public/2020/11/27/20201127左右的总结/index.html","1392264c10250174410a5835389d7f2d"],["E:/GitHubBlog/public/2020/12/23/第一个深度学习模型（pytorch）/index.html","798bfa8a38e5c6c0bbbcea68e802fd3b"],["E:/GitHubBlog/public/2021/01/02/鲍鱼最终版/index.html","9147df7565f2672395eae63cddadb835"],["E:/GitHubBlog/public/2021/01/08/GSA_NumericalExperiment/index.html","5c96fbe5b641638c67c49d864dd0a864"],["E:/GitHubBlog/public/2021/01/16/20210115-01总结/index.html","8c11cb1cdbfdb0880d5e7a404dee2ef3"],["E:/GitHubBlog/public/2021/01/17/20200117-0118总结——OpenNMT阅读（二）/index.html","04ad2c8123e649674c3888bf69a89b78"],["E:/GitHubBlog/public/2021/01/17/OpenNMT源码解析/index.html","00332e96ff4a4ebbe3d29c6e65a02593"],["E:/GitHubBlog/public/2021/01/19/重学数据结构——线性表/index.html","f048c2e1f7464d85682b99118b19c7aa"],["E:/GitHubBlog/public/2021/01/22/《深度学习入门-基于python的理论与实现》——第二章 感知机/index.html","ad1f11b986182764b3b7c680dac4f5d2"],["E:/GitHubBlog/public/2021/01/26/title复现中的问题/index.html","373849753535bd22ff0b6e1ada4717b7"],["E:/GitHubBlog/public/2021/01/26/简单链表题/index.html","89b33abf3159d62849113c8ce227ba82"],["E:/GitHubBlog/public/2021/01/28/DeepKeyphraseGeneration/index.html","c0dfab0d9c6f88a52a6898c1ac810e50"],["E:/GitHubBlog/public/2021/02/01/title模型详解/index.html","88dc2b185cadd45ec65241a43e0c97f7"],["E:/GitHubBlog/public/2021/02/14/中等链表题/index.html","1f98bd799fc1e16d97f432e9a991a96c"],["E:/GitHubBlog/public/2021/02/21/Teacher forcing in RNN/index.html","9db63b153029c4c231c07e08a01e8722"],["E:/GitHubBlog/public/2021/03/11/CopyNet代码消化记录/index.html","0fc6a979f866afc83eaecf8506f60134"],["E:/GitHubBlog/public/2021/03/11/王道机试指南习题/index.html","dbded382f2a43f246bc8580c97cceddd"],["E:/GitHubBlog/public/2021/03/13/野生kpg代码挣扎记录/index.html","77d138c060fa9c4e62e8993cb4cc0ce3"],["E:/GitHubBlog/public/2021/03/19/AcWing代码记录/index.html","c5de5c3b87939cec120524bdaf1c1925"],["E:/GitHubBlog/public/2021/03/20/2021-03-26-week3组会/index.html","d72b815ee21c90d9730b1bc26e8d4009"],["E:/GitHubBlog/public/2021/03/22/NEURAL MACHINE TRANSLATION模型/index.html","435159515a2b084ddac6aae9c3d9d7e9"],["E:/GitHubBlog/public/2021/03/23/西瓜书学习——第一章 绪论/index.html","af39f4fd24b3344820424956e069530e"],["E:/GitHubBlog/public/2021/03/24/西瓜书南瓜书 第二章/index.html","a50fd92ca8e64dcd9dd80f99d4b59803"],["E:/GitHubBlog/public/2021/04/01/野生kpg代码挣扎记录2/index.html","68adf233a11dc8a9b63e15d8d0b88e16"],["E:/GitHubBlog/public/2021/04/04/python爬虫爬取新浪财经个股新闻研报/index.html","3716f041b579d2c27773670e1555ec33"],["E:/GitHubBlog/public/2021/04/04/西瓜书南瓜书 第三章/index.html","34248a23370f26bb5a79a58ea24e1d27"],["E:/GitHubBlog/public/2021/04/09/ip被封了怎么办/index.html","053f4b71ae41d1335bca6a4d44ce360a"],["E:/GitHubBlog/public/2021/04/11/20210409-0416一周总结/index.html","8d3e8347044ae55a887b1456e82e6370"],["E:/GitHubBlog/public/2021/04/12/改代码的艰难过程/index.html","65001755a8ca1f0b194e719d031e77b9"],["E:/GitHubBlog/public/2021/04/12/数据分析（机器学习）作业/index.html","69c0a10dc8c79d7eba24d1f2fd3fd836"],["E:/GitHubBlog/public/2021/04/15/量化文本的方法/index.html","5a981166800a36a5feda821669e888ee"],["E:/GitHubBlog/public/archives/2020/01/index.html","787ef4c599dc72828c9408a664beba6b"],["E:/GitHubBlog/public/archives/2020/02/index.html","db3c80cd1b399eadfe91d0059b790902"],["E:/GitHubBlog/public/archives/2020/03/index.html","7bc99dc9ee749fab47d9a1a863cde9a9"],["E:/GitHubBlog/public/archives/2020/04/index.html","db61c075dbfe95492cebdcb976138b82"],["E:/GitHubBlog/public/archives/2020/05/index.html","f1dfc3d2dbf820ff589788fd2355feff"],["E:/GitHubBlog/public/archives/2020/07/index.html","d6c2cdd1188ae584fbc956ddfd0b7708"],["E:/GitHubBlog/public/archives/2020/08/index.html","2f5312713729e49ff24eb96cda331d5b"],["E:/GitHubBlog/public/archives/2020/09/index.html","a785c969c3806b8597fa7863d8313721"],["E:/GitHubBlog/public/archives/2020/10/index.html","35d58f7dcef2374e54e89443ecf70ecf"],["E:/GitHubBlog/public/archives/2020/11/index.html","888ef2769b95a7cab949bfb1afd67436"],["E:/GitHubBlog/public/archives/2020/12/index.html","f300bdae37e27b56e59264432849f4ee"],["E:/GitHubBlog/public/archives/2020/index.html","71f56e26f22dd3709bdee57e98ef81ad"],["E:/GitHubBlog/public/archives/2020/page/2/index.html","a022a580cd0fd00f3687c9bcfe1112c8"],["E:/GitHubBlog/public/archives/2020/page/3/index.html","631905dbcec9c8143ff4344e36930d00"],["E:/GitHubBlog/public/archives/2020/page/4/index.html","b43a7dc5a7fe9a8f5f92b0f601f08ba0"],["E:/GitHubBlog/public/archives/2020/page/5/index.html","7d7dece2cadf9bf7b347b51536e36291"],["E:/GitHubBlog/public/archives/2021/01/index.html","bba875975946edc90daf06382cc8ee34"],["E:/GitHubBlog/public/archives/2021/02/index.html","a32d37f58e5b11bae35c755c6ea83921"],["E:/GitHubBlog/public/archives/2021/03/index.html","153fc97698a4d0f8cec1040fa7827981"],["E:/GitHubBlog/public/archives/2021/04/index.html","c5d333633166b2081b25235538ab1abe"],["E:/GitHubBlog/public/archives/2021/index.html","d0b09caa1c99c39b91184c30c7cee525"],["E:/GitHubBlog/public/archives/2021/page/2/index.html","831afaf015599eb129134b868d01724e"],["E:/GitHubBlog/public/archives/2021/page/3/index.html","2185a1eec461481cc4bad0d4b7bc7891"],["E:/GitHubBlog/public/archives/index.html","aae71819bb9061dded280b94f4eaaa3c"],["E:/GitHubBlog/public/archives/page/2/index.html","f17a3ba3c10138bad7f04b059e4d6f8b"],["E:/GitHubBlog/public/archives/page/3/index.html","6391300407588dffc3cf5b61cb6515fb"],["E:/GitHubBlog/public/archives/page/4/index.html","3bc653e8033b9e99cdd12a504eb17fc0"],["E:/GitHubBlog/public/archives/page/5/index.html","175dc6e972262617d54f23290097da9d"],["E:/GitHubBlog/public/archives/page/6/index.html","dceab5041d1c4cb9b13b781f87f49f03"],["E:/GitHubBlog/public/archives/page/7/index.html","b2d9729477fedc23cedf7cf5565db7d5"],["E:/GitHubBlog/public/archives/page/8/index.html","74b939ef53b1e864382d3d521f92d459"],["E:/GitHubBlog/public/assets/css/APlayer.min.css","fbe994054426fadb2dff69d824c5c67a"],["E:/GitHubBlog/public/assets/js/APlayer.min.js","8f1017e7a73737e631ff95fa51e4e7d7"],["E:/GitHubBlog/public/assets/js/Meting.min.js","bfac0368480fd344282ec018d28f173d"],["E:/GitHubBlog/public/css/404.css","b1bb50e1fabe41adcec483f6427fb3aa"],["E:/GitHubBlog/public/css/index.css","2389364df340948a5b822422956c2c87"],["E:/GitHubBlog/public/css/var.css","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/img/algolia.svg","fd40b88ac5370a5353a50b8175c1f367"],["E:/GitHubBlog/public/img/avatar.png","6cc4a809d23e3d8946a299ae4ce4e4cd"],["E:/GitHubBlog/public/index.html","50e9f95a64b20b7cc26d6b35ee6c0abb"],["E:/GitHubBlog/public/js/copy.js","45aae54bf2e43ac27ecc41eb5e0bacf7"],["E:/GitHubBlog/public/js/fancybox.js","f84d626654b9bbc05720952b3effe062"],["E:/GitHubBlog/public/js/fireworks.js","35933ce61c158ef9c33b5e089fe757ac"],["E:/GitHubBlog/public/js/head.js","347edd99f8e3921b45fa617e839d8182"],["E:/GitHubBlog/public/js/hexo-theme-melody.js","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/js/katex.js","d971ba8b8dee1120ef66744b89cfd2b1"],["E:/GitHubBlog/public/js/scroll.js","a12ad7e37b9e325f8da3d0523d3757c7"],["E:/GitHubBlog/public/js/search/algolia.js","53160985d32d6fd66d98aa0e05b081ac"],["E:/GitHubBlog/public/js/search/local-search.js","1565b508bd866ed6fbd724a3858af5d8"],["E:/GitHubBlog/public/js/sidebar.js","d24db780974e661198eeb2e45f20a28f"],["E:/GitHubBlog/public/js/third-party/anime.min.js","9b4bbe6deb700e1c3606eab732f5eea5"],["E:/GitHubBlog/public/js/third-party/canvas-ribbon.js","09c181544ddff1db701db02ac30453e0"],["E:/GitHubBlog/public/js/third-party/jquery.fancybox.min.js","3c9fa1c1199cd4f874d855ecb1641335"],["E:/GitHubBlog/public/js/third-party/jquery.min.js","c9f5aeeca3ad37bf2aa006139b935f0a"],["E:/GitHubBlog/public/js/third-party/reveal/head.min.js","aad121203010122e05f1766d92385214"],["E:/GitHubBlog/public/js/third-party/reveal/highlight.min.js","44594243bec43813a16371af8fe7e105"],["E:/GitHubBlog/public/js/third-party/reveal/markdown.min.js","7ec4cef5a7fe3f0bf0eb4dc6d7bca114"],["E:/GitHubBlog/public/js/third-party/reveal/marked.min.js","c2a88705e206d71dc21fdc4445349127"],["E:/GitHubBlog/public/js/third-party/reveal/math.min.js","0a278fee2e57c530ab07f7d2d9ea8d96"],["E:/GitHubBlog/public/js/third-party/reveal/notes.min.js","89a0dfae4d706f9c75b317f686c3aa14"],["E:/GitHubBlog/public/js/third-party/reveal/reveal.min.js","8988419d67efb5fe93e291a357e26ec9"],["E:/GitHubBlog/public/js/third-party/reveal/zoom.min.js","9791f96e63e7d534cba2b67d4bda0419"],["E:/GitHubBlog/public/js/third-party/velocity.min.js","64da069aba987ea0512cf610600a56d1"],["E:/GitHubBlog/public/js/third-party/velocity.ui.min.js","c8ca438424a080620f7b2f4ee4b0fff1"],["E:/GitHubBlog/public/js/transition.js","911db4268f0f6621073afcced9e1bfef"],["E:/GitHubBlog/public/js/utils.js","3ff3423d966a1c351e9867813b3f6d36"],["E:/GitHubBlog/public/page/2/index.html","76cc9cc6225e2fa0c74d8c9d29918fa2"],["E:/GitHubBlog/public/page/3/index.html","4034f70d6cdf6d015d69faa38c362445"],["E:/GitHubBlog/public/page/4/index.html","4618c2096a05fe3ece940dadf70b007b"],["E:/GitHubBlog/public/page/5/index.html","734180325956ec9dd8a4e918d88d0778"],["E:/GitHubBlog/public/page/6/index.html","548a0a5b3872ef3498ab29551de57dff"],["E:/GitHubBlog/public/page/7/index.html","c63af3e131c88d1e66e8e07a728a452d"],["E:/GitHubBlog/public/page/8/index.html","7b6e22f1dd8a594e9aed992e1e6a8e52"],["E:/GitHubBlog/public/tags/Android/index.html","b646286e9f75f86304583e5bb48ad904"],["E:/GitHubBlog/public/tags/NLP/index.html","9381a1d0b0ca2ac3c7a4d1047089ab04"],["E:/GitHubBlog/public/tags/NLP/page/2/index.html","cccbe35ed5de947b83f2064978a284d2"],["E:/GitHubBlog/public/tags/NLP/page/3/index.html","56d1fa1c233677c58522d263934c9d90"],["E:/GitHubBlog/public/tags/R/index.html","0e60beaf65ceb6086866e62c983a8609"],["E:/GitHubBlog/public/tags/index.html","95b9d7108ae9425629af37269aae9838"],["E:/GitHubBlog/public/tags/java/index.html","e840d1862e80429efeb18f40a56d6bbd"],["E:/GitHubBlog/public/tags/java/page/2/index.html","0212e932b39ae16b11602f94cb2184ed"],["E:/GitHubBlog/public/tags/leetcode/index.html","42ed601b912e0e9b3c81cd8bff8cc93c"],["E:/GitHubBlog/public/tags/python/index.html","85ed66fe285d1169069969599619c877"],["E:/GitHubBlog/public/tags/pytorch/index.html","552039918bba39945ee7c84611b8acf0"],["E:/GitHubBlog/public/tags/代码/index.html","c22b6a5c6948d76cb62d11568f7e7d9d"],["E:/GitHubBlog/public/tags/优化方法/index.html","dd85b3ae0b96026dcd352c82668eb346"],["E:/GitHubBlog/public/tags/总结/index.html","da909eca1c24bcbf51f8ea40c52d2038"],["E:/GitHubBlog/public/tags/总结/page/2/index.html","ef1befeb3be50fb69f76eae8a79a35df"],["E:/GitHubBlog/public/tags/数据分析/index.html","24f4be641e521c495a46b2f5599baf10"],["E:/GitHubBlog/public/tags/数据挖掘/index.html","827095b9ec40a28fb2cf837dbcefb7cd"],["E:/GitHubBlog/public/tags/数据科学实战/index.html","38e2a1ac3eeace0f58dd28b65602078c"],["E:/GitHubBlog/public/tags/数据结构/index.html","559b7bbe95b8e40c5a2249a9c9a1c436"],["E:/GitHubBlog/public/tags/机器学习/index.html","7b2a5aa496d2b2e089a0e9f65181f590"],["E:/GitHubBlog/public/tags/深度学习/index.html","59570858e7f37813b95516a8f1ac4cba"],["E:/GitHubBlog/public/tags/爬虫/index.html","fece638656e95c201bc58d8f54955252"],["E:/GitHubBlog/public/tags/笔记/index.html","f809e1a76ca568a4befcca49619e04b2"],["E:/GitHubBlog/public/tags/论文/index.html","0ce7b23af64f58e0d45fc7c6536527a8"],["E:/GitHubBlog/public/tags/论文/page/2/index.html","0e2561e5d00e5a7c7cc51dc2b1997bdf"],["E:/GitHubBlog/public/tags/论文/page/3/index.html","99a2922523a05fb747b8466621cfcd4a"],["E:/GitHubBlog/public/tags/读书笔记/index.html","e3873c4ca1134eb29ecb47c55a1b6c17"],["E:/GitHubBlog/public/tags/量化交易/index.html","288819ebf008dd43dfdf63a0119eb4cd"]];
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







