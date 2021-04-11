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

var precacheConfig = [["E:/GitHubBlog/public/2020/01/28/hello-world/index.html","ded6ec2478224f243619e66cc67c1fdc"],["E:/GitHubBlog/public/2020/02/08/python爬虫学习总结/index.html","76c5838a37b3a68b0a391974fc5e7072"],["E:/GitHubBlog/public/2020/02/19/数据结构第一周-绪论/index.html","95a9ffa7cddf2a33fffaf21a2ccc107d"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-1/index.html","fb12cc5aa623852106544a628e5f4c88"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-2-创建数据集/index.html","0ec6b8fe57ffbb6c05c3770de7ae0148"],["E:/GitHubBlog/public/2020/03/01/《R语言实战》学习总结-3-图形初阶/index.html","423f9ce6c03aa87a88982d9110a8f120"],["E:/GitHubBlog/public/2020/03/02/数据结构第二周-线性结构1/index.html","c832816a31c0d03fe4e514ff13f43d07"],["E:/GitHubBlog/public/2020/03/03/《R语言实战》学习总结-4-基本数据管理/index.html","898a8fbe4b123cc0b515e02302fcaf8d"],["E:/GitHubBlog/public/2020/03/14/数据结构第三周-线性结构2/index.html","9437eaa6cd4c7cf67a8f8cfcdac54d18"],["E:/GitHubBlog/public/2020/03/14/数据结构第四周-栈与队列/index.html","959bdb0d5078f635a834562537ecd72e"],["E:/GitHubBlog/public/2020/03/17/leetcode记录-1、88、20/index.html","de9000b705ca688a19c1cddd0d657794"],["E:/GitHubBlog/public/2020/03/20/温度转换程序/index.html","04be5a84d26a1c8196879a315788ba66"],["E:/GitHubBlog/public/2020/03/22/数据结构第五周-递归/index.html","18382c9bac86c9d430c7e7130546f362"],["E:/GitHubBlog/public/2020/03/22/数据结构第六周-串与数组/index.html","770bf15af0f6370ed2c368765b407f71"],["E:/GitHubBlog/public/2020/04/02/leetcode记录-52/index.html","858375b14cb6c14ed72fcf3b10bee069"],["E:/GitHubBlog/public/2020/04/17/Android第八周作业-每天第一次打开应用更新数据/index.html","f1be88599c7757d5078e6d5e6349bbaa"],["E:/GitHubBlog/public/2020/05/21/attention模型/index.html","a33440db3319ca30ff4c796c0940c079"],["E:/GitHubBlog/public/2020/05/21/编码器与解码器框架/index.html","57b13888676bb92552b4beeffdba55c4"],["E:/GitHubBlog/public/2020/07/11/思考，快与慢/index.html","a8f036eda157941f55565a2a64ec4c06"],["E:/GitHubBlog/public/2020/07/13/20200713-0717总结/index.html","80254b7ee1a91f71c5251686230ea3c7"],["E:/GitHubBlog/public/2020/07/13/ZeroShotCeres Zero-Shot Relation Extraction/index.html","708f567a396b34669629c85df4e1e16a"],["E:/GitHubBlog/public/2020/07/20/Keyphrase Generation A Text Summarization Struggle/index.html","82482095b8e6ac3872f96e814162de97"],["E:/GitHubBlog/public/2020/07/20/TPR/index.html","e53d975df425784c4f20c3462db6037e"],["E:/GitHubBlog/public/2020/07/23/Keyphrase Generation for Scientific Document Retrieval/index.html","78e01eaa36ed8dea5921989c0814f34a"],["E:/GitHubBlog/public/2020/07/24/20200718-0724总结/index.html","f405a2d99057714ac4208cc5b3a05258"],["E:/GitHubBlog/public/2020/07/31/20200725-0731总结/index.html","7081055b8f47eb65b1423f32bb752e52"],["E:/GitHubBlog/public/2020/08/07/20200801-0807总结/index.html","2c39fbdf4417914859f1bc65c8407968"],["E:/GitHubBlog/public/2020/08/13/李宏毅-机器学习21/index.html","46cccc55426d8e313a0b622b5545a99d"],["E:/GitHubBlog/public/2020/08/14/20200808-0815总结/index.html","cfa11b2a637deb7067f10a51610895c1"],["E:/GitHubBlog/public/2020/09/03/20200816-0903总结/index.html","181c9b8e2dc84b466f997191846098e9"],["E:/GitHubBlog/public/2020/09/16/20200914-0918总结/index.html","655171bdfe59d4ca3a5c453b011c7d9c"],["E:/GitHubBlog/public/2020/09/25/20200919-0925总结/index.html","04883f0c659bde6522f8695c56d1dcdd"],["E:/GitHubBlog/public/2020/10/06/《Deep Keyphrase Generation》详细解析！/index.html","4bb5112a5069ce104b50fb3026cb5774"],["E:/GitHubBlog/public/2020/10/16/20201008-1017总结/index.html","f8f504e7b446b99a16b1dded9c7432e5"],["E:/GitHubBlog/public/2020/10/20/特征选择的几种方法(一)————Filter过滤法/index.html","241e3f825b4cb449e53ac9941a6226a5"],["E:/GitHubBlog/public/2020/10/21/PCA/index.html","99e77ecf8eae0520894015b470cf66dd"],["E:/GitHubBlog/public/2020/10/22/20201018-1023总结/index.html","7eb072cd00f32f971d5e7071ed4a48ea"],["E:/GitHubBlog/public/2020/10/24/20201024-1030/index.html","5616b1bb2c18bb9a4cc6f9818a200ad5"],["E:/GitHubBlog/public/2020/10/30/1030讨论记录/index.html","81339157b692b638bd68477326dd4138"],["E:/GitHubBlog/public/2020/11/03/20201103-总结/index.html","d94c70603c104a11ac66cc206cecc847"],["E:/GitHubBlog/public/2020/11/05/线性回归与逻辑回归/index.html","238209dadf0c9a6e7126adecdc5d7f32"],["E:/GitHubBlog/public/2020/11/07/决策树/index.html","14790158f3786878ed3e69ca51853a31"],["E:/GitHubBlog/public/2020/11/13/20201107-1113总结/index.html","d92286e16d479a1bde48d22dfa85b12f"],["E:/GitHubBlog/public/2020/11/15/20201114-119总结/index.html","e7a0250cf9b57a0d6a0736b4a3d7dda4"],["E:/GitHubBlog/public/2020/11/16/手推一个SVM/index.html","da60584ceedfab036017e563d7f08567"],["E:/GitHubBlog/public/2020/11/27/20201127左右的总结/index.html","f02046900bde335f99c5cd6c6c88471a"],["E:/GitHubBlog/public/2020/12/23/第一个深度学习模型（pytorch）/index.html","376bad527544756b7f25b413765660de"],["E:/GitHubBlog/public/2021/01/02/鲍鱼最终版/index.html","e9cf247ee21f59d0155ae869fb4bc530"],["E:/GitHubBlog/public/2021/01/08/GSA_NumericalExperiment/index.html","f770a2b1c63906342dc0191068ba902d"],["E:/GitHubBlog/public/2021/01/16/20210115-01总结/index.html","31edaa9e0312950a7458724968495c8c"],["E:/GitHubBlog/public/2021/01/17/20200117-0118总结——OpenNMT阅读（二）/index.html","cc575584b57b53a44f1c83cca3ba9921"],["E:/GitHubBlog/public/2021/01/17/OpenNMT源码解析/index.html","50b54185c54520b0acddead7bb8857f2"],["E:/GitHubBlog/public/2021/01/19/重学数据结构——线性表/index.html","dbef9f273f9cad4e64edf84546641c94"],["E:/GitHubBlog/public/2021/01/22/《深度学习入门-基于python的理论与实现》——第二章 感知机/index.html","a352ea5fec50d65e4e5043c8eb2bbaef"],["E:/GitHubBlog/public/2021/01/26/title复现中的问题/index.html","59c9b0707c2cb2760e0fea7182b6e53e"],["E:/GitHubBlog/public/2021/01/26/简单链表题/index.html","a531b8840d25ef318e9500f7f9e94c66"],["E:/GitHubBlog/public/2021/01/28/DeepKeyphraseGeneration/index.html","ef1f0e18d53fe0ea81f6d5cc2356a55a"],["E:/GitHubBlog/public/2021/02/01/title模型详解/index.html","0de4bb02fdac6af5c80274e68a509d06"],["E:/GitHubBlog/public/2021/02/14/中等链表题/index.html","da345cdc4684eaa215ef4dc071330da4"],["E:/GitHubBlog/public/2021/02/21/Teacher forcing in RNN/index.html","738c0d93275eb5be26c0ddae222f75cf"],["E:/GitHubBlog/public/2021/03/11/CopyNet代码消化记录/index.html","04424b623bf32371229484b133f129d2"],["E:/GitHubBlog/public/2021/03/11/王道机试指南习题/index.html","5e6e1b72d105cd65a87531a3f81b3e11"],["E:/GitHubBlog/public/2021/03/13/野生kpg代码挣扎记录/index.html","0f50cc691266953f1538c3c7c7940b64"],["E:/GitHubBlog/public/2021/03/19/AcWing代码记录/index.html","07b64957e75074fb1481b469f195140a"],["E:/GitHubBlog/public/2021/03/20/2021-03-26-week3组会/index.html","5218c0f283912b42ce449ccdcd82c56e"],["E:/GitHubBlog/public/2021/03/22/NEURAL MACHINE TRANSLATION模型/index.html","78bb834f2134752a5699e6640fa88a5e"],["E:/GitHubBlog/public/2021/03/23/西瓜书学习——第一章 绪论/index.html","e1056ed3a861cb0db1be8605d8a198cb"],["E:/GitHubBlog/public/2021/03/24/西瓜书南瓜书 第二章/index.html","16fdb3ee0ba3574e09d59170dfc516ec"],["E:/GitHubBlog/public/2021/04/01/野生kpg代码挣扎记录2/index.html","9d2d30b297ffaad75148a5f88ce57209"],["E:/GitHubBlog/public/2021/04/04/python爬虫爬取新浪财经个股新闻研报/index.html","c4a040b3517818fd258e8df48493034b"],["E:/GitHubBlog/public/2021/04/04/西瓜书南瓜书 第三章/index.html","407644d686526d8e7c28ad8eb874e1f8"],["E:/GitHubBlog/public/2021/04/09/ip被封了怎么办/index.html","e508c91c290818888a0a2a61bd017c85"],["E:/GitHubBlog/public/2021/04/11/20210409-0416一周总结/index.html","d20f6e01081774b9dc3c04a2ed5981d7"],["E:/GitHubBlog/public/2021/04/12/数据分析（机器学习）作业/index.html","5279dc42cc99f676e36dbfc49e0846df"],["E:/GitHubBlog/public/archives/2020/01/index.html","c24164eb6621563196da3a6157c0c8dc"],["E:/GitHubBlog/public/archives/2020/02/index.html","235bc6632fc714f8d7efd1d5f5fccc21"],["E:/GitHubBlog/public/archives/2020/03/index.html","9ececbbcd3fe07715e61aa29d5cc99d4"],["E:/GitHubBlog/public/archives/2020/04/index.html","3ea483042cf406141b752ddab1105a36"],["E:/GitHubBlog/public/archives/2020/05/index.html","c4eec4016e3864814fad16124e19b3f2"],["E:/GitHubBlog/public/archives/2020/07/index.html","8480944b97574d526065ca6aa251b7e7"],["E:/GitHubBlog/public/archives/2020/08/index.html","ee67fe28e502cdd2f0b24717dd5636c7"],["E:/GitHubBlog/public/archives/2020/09/index.html","bb827e0c813cfef59362b4c3f69db19c"],["E:/GitHubBlog/public/archives/2020/10/index.html","a45a738ef468a39aef56d1a6390a571b"],["E:/GitHubBlog/public/archives/2020/11/index.html","e9bae0f400292ac2c22fa5f4e31ecfbc"],["E:/GitHubBlog/public/archives/2020/12/index.html","bed2982652e21935d2d28cee743db079"],["E:/GitHubBlog/public/archives/2020/index.html","6724a494b698b7bdf47fbcf7078cee24"],["E:/GitHubBlog/public/archives/2020/page/2/index.html","b6b5a4617ebd86da5e7e08000029fe77"],["E:/GitHubBlog/public/archives/2020/page/3/index.html","3359bd49e2e0266e52d98384ce4c9965"],["E:/GitHubBlog/public/archives/2020/page/4/index.html","abedc1c6a272e1d7de027d834a98dce5"],["E:/GitHubBlog/public/archives/2020/page/5/index.html","a20702e1368e05e537d654bf43ccdd6b"],["E:/GitHubBlog/public/archives/2021/01/index.html","456a1b78206b6c3cae0b98f86d3fe341"],["E:/GitHubBlog/public/archives/2021/02/index.html","a02e886b2d6ddfaa3893374d77aac8cf"],["E:/GitHubBlog/public/archives/2021/03/index.html","7a4bba593d4f9c3ec7bef7ce5287c001"],["E:/GitHubBlog/public/archives/2021/04/index.html","fb4c9e402e172f9f402b16279a5fa353"],["E:/GitHubBlog/public/archives/2021/index.html","e43cd75eb6b712914a4f9bfbcad63795"],["E:/GitHubBlog/public/archives/2021/page/2/index.html","287a3624b8a109c7d7e2fa069f514acd"],["E:/GitHubBlog/public/archives/2021/page/3/index.html","55a90232cea9dfd74a85f2abde9ad944"],["E:/GitHubBlog/public/archives/index.html","5c3a61e365a5532de00d9912a98c52d6"],["E:/GitHubBlog/public/archives/page/2/index.html","96720e3c303c26e92decb949f16c31f9"],["E:/GitHubBlog/public/archives/page/3/index.html","1b263b8cfcd031ef500c940a47387ce5"],["E:/GitHubBlog/public/archives/page/4/index.html","8f7e378a82a8027d98074b999bfc27fd"],["E:/GitHubBlog/public/archives/page/5/index.html","036ef7927f8cd0cda51f3c451a4cffe0"],["E:/GitHubBlog/public/archives/page/6/index.html","97c281f14d92031f3dcd6379dba67afd"],["E:/GitHubBlog/public/archives/page/7/index.html","e4f31b5f0e96e4720af3a901bd6d7c77"],["E:/GitHubBlog/public/archives/page/8/index.html","55e8f3844fb60950b716d9972ea8d937"],["E:/GitHubBlog/public/assets/css/APlayer.min.css","fbe994054426fadb2dff69d824c5c67a"],["E:/GitHubBlog/public/assets/js/APlayer.min.js","8f1017e7a73737e631ff95fa51e4e7d7"],["E:/GitHubBlog/public/assets/js/Meting.min.js","bfac0368480fd344282ec018d28f173d"],["E:/GitHubBlog/public/css/404.css","b1bb50e1fabe41adcec483f6427fb3aa"],["E:/GitHubBlog/public/css/index.css","2389364df340948a5b822422956c2c87"],["E:/GitHubBlog/public/css/var.css","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/img/algolia.svg","fd40b88ac5370a5353a50b8175c1f367"],["E:/GitHubBlog/public/img/avatar.png","6cc4a809d23e3d8946a299ae4ce4e4cd"],["E:/GitHubBlog/public/index.html","9b0e21fbed93332749a2ef5727ea00de"],["E:/GitHubBlog/public/js/copy.js","45aae54bf2e43ac27ecc41eb5e0bacf7"],["E:/GitHubBlog/public/js/fancybox.js","f84d626654b9bbc05720952b3effe062"],["E:/GitHubBlog/public/js/fireworks.js","35933ce61c158ef9c33b5e089fe757ac"],["E:/GitHubBlog/public/js/head.js","347edd99f8e3921b45fa617e839d8182"],["E:/GitHubBlog/public/js/hexo-theme-melody.js","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/js/katex.js","d971ba8b8dee1120ef66744b89cfd2b1"],["E:/GitHubBlog/public/js/scroll.js","a12ad7e37b9e325f8da3d0523d3757c7"],["E:/GitHubBlog/public/js/search/algolia.js","53160985d32d6fd66d98aa0e05b081ac"],["E:/GitHubBlog/public/js/search/local-search.js","1565b508bd866ed6fbd724a3858af5d8"],["E:/GitHubBlog/public/js/sidebar.js","d24db780974e661198eeb2e45f20a28f"],["E:/GitHubBlog/public/js/third-party/anime.min.js","9b4bbe6deb700e1c3606eab732f5eea5"],["E:/GitHubBlog/public/js/third-party/canvas-ribbon.js","09c181544ddff1db701db02ac30453e0"],["E:/GitHubBlog/public/js/third-party/jquery.fancybox.min.js","3c9fa1c1199cd4f874d855ecb1641335"],["E:/GitHubBlog/public/js/third-party/jquery.min.js","c9f5aeeca3ad37bf2aa006139b935f0a"],["E:/GitHubBlog/public/js/third-party/reveal/head.min.js","aad121203010122e05f1766d92385214"],["E:/GitHubBlog/public/js/third-party/reveal/highlight.min.js","44594243bec43813a16371af8fe7e105"],["E:/GitHubBlog/public/js/third-party/reveal/markdown.min.js","7ec4cef5a7fe3f0bf0eb4dc6d7bca114"],["E:/GitHubBlog/public/js/third-party/reveal/marked.min.js","c2a88705e206d71dc21fdc4445349127"],["E:/GitHubBlog/public/js/third-party/reveal/math.min.js","0a278fee2e57c530ab07f7d2d9ea8d96"],["E:/GitHubBlog/public/js/third-party/reveal/notes.min.js","89a0dfae4d706f9c75b317f686c3aa14"],["E:/GitHubBlog/public/js/third-party/reveal/reveal.min.js","8988419d67efb5fe93e291a357e26ec9"],["E:/GitHubBlog/public/js/third-party/reveal/zoom.min.js","9791f96e63e7d534cba2b67d4bda0419"],["E:/GitHubBlog/public/js/third-party/velocity.min.js","64da069aba987ea0512cf610600a56d1"],["E:/GitHubBlog/public/js/third-party/velocity.ui.min.js","c8ca438424a080620f7b2f4ee4b0fff1"],["E:/GitHubBlog/public/js/transition.js","911db4268f0f6621073afcced9e1bfef"],["E:/GitHubBlog/public/js/utils.js","3ff3423d966a1c351e9867813b3f6d36"],["E:/GitHubBlog/public/page/2/index.html","a675d729a6d2f251c4a33574223316d3"],["E:/GitHubBlog/public/page/3/index.html","4b64a3647520a394d2b968a25b8adfef"],["E:/GitHubBlog/public/page/4/index.html","899973f3a9841722cc4930c7c255d09e"],["E:/GitHubBlog/public/page/5/index.html","6e7d87f2314f2ca74d41022b6dfd06e6"],["E:/GitHubBlog/public/page/6/index.html","564c455db1f2be48b6a7ada9ce4e207b"],["E:/GitHubBlog/public/page/7/index.html","0e56ec1e0f8b062342cf2ff3e9203fa7"],["E:/GitHubBlog/public/page/8/index.html","d9cb8d6d40fb48f9daf8f4f6e033a447"],["E:/GitHubBlog/public/tags/Android/index.html","bc4f8bf80ec6a68d465123c5c93984fb"],["E:/GitHubBlog/public/tags/NLP/index.html","14c74d74b583e6791dfa050879e2e313"],["E:/GitHubBlog/public/tags/NLP/page/2/index.html","50049b5ed2c06cf27c80c030d143a245"],["E:/GitHubBlog/public/tags/NLP/page/3/index.html","b140aa36e10c1f5e5e1c435c097c4908"],["E:/GitHubBlog/public/tags/R/index.html","fb2f932bf84005662271bec37ba3ae63"],["E:/GitHubBlog/public/tags/index.html","8fe982e13007bbbe6e4e29f8fb7fca2f"],["E:/GitHubBlog/public/tags/java/index.html","be165aae35d78bbf1a804d8aded1f45a"],["E:/GitHubBlog/public/tags/java/page/2/index.html","5ffb592a2713d8f90db95b3c8589a0fc"],["E:/GitHubBlog/public/tags/leetcode/index.html","379e0461ea4c5e0c88044bcd7a30d76f"],["E:/GitHubBlog/public/tags/python/index.html","f653625f1d500d11c1a3814ff62d7584"],["E:/GitHubBlog/public/tags/pytorch/index.html","e2860f43fa5bfa47f1e4e72e58d622d7"],["E:/GitHubBlog/public/tags/代码/index.html","bf7c4965526ccfce870b34997bd7d8ad"],["E:/GitHubBlog/public/tags/优化方法/index.html","163dd2b0eeedde57a78627f5184579f7"],["E:/GitHubBlog/public/tags/总结/index.html","2de1283541357d32f73736f81cfa9c4f"],["E:/GitHubBlog/public/tags/总结/page/2/index.html","9b5e26e56389b3d8159a218153c1f6aa"],["E:/GitHubBlog/public/tags/数据分析/index.html","6b3b266d254b66e9f201b9261147213a"],["E:/GitHubBlog/public/tags/数据挖掘/index.html","4d077405ef452ed9cdd669b308f10800"],["E:/GitHubBlog/public/tags/数据结构/index.html","cb41c29a0c98e6cb27d19622af27fcad"],["E:/GitHubBlog/public/tags/机器学习/index.html","34aa451f4c9147f089fbdf2a94395280"],["E:/GitHubBlog/public/tags/深度学习/index.html","766724f39dd4b31c9b22de9ae0b13bd0"],["E:/GitHubBlog/public/tags/爬虫/index.html","bfbf755f8bb12ff6b57b987babb8db0a"],["E:/GitHubBlog/public/tags/笔记/index.html","a35da8247f56b3a898d0c3eace3dd2d6"],["E:/GitHubBlog/public/tags/论文/index.html","3f746e05ee84a0866a2c5c593979847b"],["E:/GitHubBlog/public/tags/论文/page/2/index.html","edf65cbe26c46ca4e7382eb4f2a5cb24"],["E:/GitHubBlog/public/tags/论文/page/3/index.html","5b1964c48543c474af45d2f48a627662"],["E:/GitHubBlog/public/tags/读书笔记/index.html","8a1da1c5d66a5462df55cf7e3c2a7cfa"],["E:/GitHubBlog/public/tags/量化交易/index.html","5d71e65dadf3adbaa540621db6534458"]];
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







