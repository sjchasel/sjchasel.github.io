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

var precacheConfig = [["E:/GitHubBlog/public/2020/01/28/hello-world/index.html","d8584718a57ef676766aa4c60185dad0"],["E:/GitHubBlog/public/2020/02/08/python爬虫学习总结/index.html","0247b533cd750552d3feb3ddcca7a028"],["E:/GitHubBlog/public/2020/02/19/数据结构第一周-绪论/index.html","16b2edd1c59de0c5d6ceb1b0ed17f29d"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-1/index.html","d6f57b4856cdc518af72f1a5af18f9ec"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-2-创建数据集/index.html","c42ff515b4a2bc9b3a4b851d06c77da8"],["E:/GitHubBlog/public/2020/03/01/《R语言实战》学习总结-3-图形初阶/index.html","e3422e5447b2cd1cfa6583cf17f13069"],["E:/GitHubBlog/public/2020/03/02/数据结构第二周-线性结构1/index.html","3055d94f9c8250a11fdb153e9f2891a0"],["E:/GitHubBlog/public/2020/03/03/《R语言实战》学习总结-4-基本数据管理/index.html","a2dfdc9f702dbb5efdce8e0f09e5d188"],["E:/GitHubBlog/public/2020/03/14/数据结构第三周-线性结构2/index.html","b009f9c6a6d4839e054c24cdfbcd5bfb"],["E:/GitHubBlog/public/2020/03/14/数据结构第四周-栈与队列/index.html","5e4aa4422b33acc4f76facf5e32ff78e"],["E:/GitHubBlog/public/2020/03/17/leetcode记录-1、88、20/index.html","7cf8d9742afa5bda1d84107870a9ccc1"],["E:/GitHubBlog/public/2020/03/20/温度转换程序/index.html","1d91125e630ea2c254cfbc29899d05b8"],["E:/GitHubBlog/public/2020/03/22/数据结构第五周-递归/index.html","675a8f57d0fd1a0cde413499e65e70d9"],["E:/GitHubBlog/public/2020/03/22/数据结构第六周-串与数组/index.html","d29b1baf428568a7ba26ace5d44018f3"],["E:/GitHubBlog/public/2020/04/02/leetcode记录-52/index.html","299a4e2fcfee5e08c3a55ec021612dc3"],["E:/GitHubBlog/public/2020/04/17/Android第八周作业-每天第一次打开应用更新数据/index.html","8bdd778e899ada025bc40f9a8ca567ba"],["E:/GitHubBlog/public/2020/05/21/attention模型/index.html","831410fd9f02856892afbc417c0f935f"],["E:/GitHubBlog/public/2020/05/21/编码器与解码器框架/index.html","dfdd3198b01609e6937186e37d348307"],["E:/GitHubBlog/public/2020/07/11/思考，快与慢/index.html","488b8014e852f36878b6301303728819"],["E:/GitHubBlog/public/2020/07/13/20200713-0717总结/index.html","822d6990c06ee0306f0b31d41be7a9c2"],["E:/GitHubBlog/public/2020/07/13/ZeroShotCeres Zero-Shot Relation Extraction/index.html","f12009702159a3ba4dffe0f2c8e1135f"],["E:/GitHubBlog/public/2020/07/20/Keyphrase Generation A Text Summarization Struggle/index.html","b869dc766400a7e0b589cda442395748"],["E:/GitHubBlog/public/2020/07/20/TPR/index.html","59e215dd6dcab29ce5a55b6b8f1cf79c"],["E:/GitHubBlog/public/2020/07/23/Keyphrase Generation for Scientific Document Retrieval/index.html","ca8be18cffb133a2574b329391603cbb"],["E:/GitHubBlog/public/2020/07/24/20200718-0724总结/index.html","047067d0ef0b2726c7600cecf5833213"],["E:/GitHubBlog/public/2020/07/31/20200725-0731总结/index.html","f229b655be33e7740bd78388edb29a3f"],["E:/GitHubBlog/public/2020/08/07/20200801-0807总结/index.html","e92346b3c2b30a43b196a63873442a6d"],["E:/GitHubBlog/public/2020/08/13/李宏毅-机器学习21/index.html","db8977e7342ad67b1d6e80fd52afaa1f"],["E:/GitHubBlog/public/2020/08/14/20200808-0815总结/index.html","f4b2fb4622948c220816e0e11f6bcfe2"],["E:/GitHubBlog/public/2020/09/03/20200816-0903总结/index.html","c93d313a5c527fbf8cc99de7c458a94a"],["E:/GitHubBlog/public/2020/09/16/20200914-0918总结/index.html","175b4bec0983ceef010891ff74e6dff0"],["E:/GitHubBlog/public/2020/09/25/20200919-0925总结/index.html","78112f4e3d839f36ae0252d63b509a4b"],["E:/GitHubBlog/public/2020/10/06/《Deep Keyphrase Generation》详细解析！/index.html","36aed303f91ff932e07115f7bf67553b"],["E:/GitHubBlog/public/2020/10/16/20201008-1017总结/index.html","5c63754f9af60282cb006fa2802e3ee2"],["E:/GitHubBlog/public/2020/10/20/特征选择的几种方法(一)————Filter过滤法/index.html","b830f92226cd8c29a71bf5ac56f01c08"],["E:/GitHubBlog/public/2020/10/21/PCA/index.html","a305b027831421ca7dd90e21671edd42"],["E:/GitHubBlog/public/2020/10/22/20201018-1023总结/index.html","19b91e6cdefa2208865bd73ea93ce1c7"],["E:/GitHubBlog/public/2020/10/24/20201024-1030/index.html","6124487ff72363f46748d976265295b3"],["E:/GitHubBlog/public/2020/10/30/1030讨论记录/index.html","610496c621433c89d14851e8d26cfa9d"],["E:/GitHubBlog/public/2020/11/03/20201103-总结/index.html","54e39818b63ac11604e71ba989d7d06a"],["E:/GitHubBlog/public/2020/11/05/线性回归与逻辑回归/index.html","8c237596cd375524a6ca6c9f9fc5c46d"],["E:/GitHubBlog/public/2020/11/07/决策树/index.html","8e3877c125a6ce59f9a255a799b124c4"],["E:/GitHubBlog/public/2020/11/13/20201107-1113总结/index.html","7685faece5285b4268d5f0913c0a23f5"],["E:/GitHubBlog/public/2020/11/15/20201114-119总结/index.html","c0c808ba660c3dffc63f1d96a6ad745d"],["E:/GitHubBlog/public/2020/11/16/手推一个SVM/index.html","35ee03f4b604def84e484ae7809b230b"],["E:/GitHubBlog/public/2020/11/27/20201127左右的总结/index.html","d9af59b7b832d6f5dd46ab0c23d9ce90"],["E:/GitHubBlog/public/2020/12/23/第一个深度学习模型（pytorch）/index.html","4c9fffbf098d3074ae485a477f5945ff"],["E:/GitHubBlog/public/2021/01/02/鲍鱼最终版/index.html","80271d6cce1f6a52d2063fbf8e518d61"],["E:/GitHubBlog/public/2021/01/08/GSA_NumericalExperiment/index.html","9663a3aa65a9b2991d5d69edfcf98ab0"],["E:/GitHubBlog/public/2021/01/16/20210115-01总结/index.html","8d9767664d313ccdc323e3efa9e18206"],["E:/GitHubBlog/public/2021/01/17/20200117-0118总结——OpenNMT阅读（二）/index.html","5b4fabbeebd732b9b97bdad2f1dd64cd"],["E:/GitHubBlog/public/2021/01/17/OpenNMT源码解析/index.html","6a0425f937419f82f9cb46872b202dc1"],["E:/GitHubBlog/public/2021/01/19/重学数据结构——线性表/index.html","92f2f14f4a696419528d3280a8a94ce1"],["E:/GitHubBlog/public/2021/01/22/《深度学习入门-基于python的理论与实现》——第二章 感知机/index.html","11fa1552d69f39e63c49ae28c8f4242b"],["E:/GitHubBlog/public/2021/01/26/title复现中的问题/index.html","026b3c34487a7fc3c3d1b463ff03d2e0"],["E:/GitHubBlog/public/2021/01/26/简单链表题/index.html","71c9f0b77458602fd9c3be0658a953f3"],["E:/GitHubBlog/public/2021/01/28/DeepKeyphraseGeneration/index.html","e17cef5f23f1129e5bd47ee44383c0c4"],["E:/GitHubBlog/public/2021/02/01/title模型详解/index.html","72fb457fb7be1c27f579708304f6c553"],["E:/GitHubBlog/public/2021/02/14/中等链表题/index.html","7f6b33c94ee4eb134debe96510a7a4b0"],["E:/GitHubBlog/public/2021/02/21/Teacher forcing in RNN/index.html","d2699b21b49bdc4965b64c4795c17d83"],["E:/GitHubBlog/public/2021/03/11/CopyNet代码消化记录/index.html","aff3dd03f2ed7029804733953eafea84"],["E:/GitHubBlog/public/2021/03/11/王道机试指南习题/index.html","337ad25965cc4185e67c883c805dfb25"],["E:/GitHubBlog/public/2021/03/13/野生kpg代码挣扎记录/index.html","2539c38bd79dc8d8300fd1205cabd645"],["E:/GitHubBlog/public/2021/03/19/AcWing代码记录/index.html","814e2adac5b915e6bbaab6ec396578f3"],["E:/GitHubBlog/public/2021/03/20/2021-03-26-week3组会/index.html","ae6eb36fcb2ef336a989022951d6ef41"],["E:/GitHubBlog/public/2021/03/22/NEURAL MACHINE TRANSLATION模型/index.html","856f3e6bf75a90187e86fd93c504dc64"],["E:/GitHubBlog/public/2021/03/23/西瓜书学习——第一章 绪论/index.html","317678b436804b639acbc814c6300768"],["E:/GitHubBlog/public/2021/03/24/西瓜书南瓜书 第二章/index.html","38dfe9b2bd56552cd350fb64c381f560"],["E:/GitHubBlog/public/2021/04/01/野生kpg代码挣扎记录2/index.html","9a89a9e154819dc8d064092f0b5bb861"],["E:/GitHubBlog/public/2021/04/04/python爬虫爬取新浪财经个股新闻研报/index.html","99f7d8d9f39d35efa0517e4090e2ad4f"],["E:/GitHubBlog/public/2021/04/04/西瓜书南瓜书 第三章/index.html","a8fe78605d069a8fc9e7515ca2f0416a"],["E:/GitHubBlog/public/2021/04/09/ip被封了怎么办/index.html","55051a96383ce68768b9faf36ba61e07"],["E:/GitHubBlog/public/archives/2020/01/index.html","2b453e21b2a6fe3b715526be277e3c5f"],["E:/GitHubBlog/public/archives/2020/02/index.html","6753b8316f510ccda808d80db1ab139a"],["E:/GitHubBlog/public/archives/2020/03/index.html","12b1482f3807c544587c8948e787882a"],["E:/GitHubBlog/public/archives/2020/04/index.html","8de484f1b245e13baad8a79b82a12be5"],["E:/GitHubBlog/public/archives/2020/05/index.html","6c9368e95a33dae1fd58b964878aaf28"],["E:/GitHubBlog/public/archives/2020/07/index.html","f488dd586fdf0dbfddd17743d8ea1a6d"],["E:/GitHubBlog/public/archives/2020/08/index.html","135456ad244ce9d53b6cc919a7b34326"],["E:/GitHubBlog/public/archives/2020/09/index.html","fa9eeee3b6a6a9bc07efd024e20abd8c"],["E:/GitHubBlog/public/archives/2020/10/index.html","db5b6ae197982a0f6d4d3dc4b5e5fc78"],["E:/GitHubBlog/public/archives/2020/11/index.html","ebe0973e926d30e6d777a3f27e799b1d"],["E:/GitHubBlog/public/archives/2020/12/index.html","592f509154dc67c988cfee9fdf566741"],["E:/GitHubBlog/public/archives/2020/index.html","d9b02ff9dbec3d5fab206d7a21fc2ffc"],["E:/GitHubBlog/public/archives/2020/page/2/index.html","5b152a625ca6dec0ddee5d55a7de030d"],["E:/GitHubBlog/public/archives/2020/page/3/index.html","ead354334708716f9bcc137c50f90995"],["E:/GitHubBlog/public/archives/2020/page/4/index.html","e2c4cd190317bd1dd6437241a51cdd44"],["E:/GitHubBlog/public/archives/2020/page/5/index.html","59e44aa49b8f231e1a57162260b2c5ca"],["E:/GitHubBlog/public/archives/2021/01/index.html","13b71351ec7c8d2d23e37fe1a0358bdb"],["E:/GitHubBlog/public/archives/2021/02/index.html","d1c8c08a23023826ec2a561f524e6a86"],["E:/GitHubBlog/public/archives/2021/03/index.html","1f17389e50a3bdfbc8ede17e33c41544"],["E:/GitHubBlog/public/archives/2021/04/index.html","ea8fbac3bcea193245b35c54faa6ccbf"],["E:/GitHubBlog/public/archives/2021/index.html","2f8b2ab973d39511493a190fe966531c"],["E:/GitHubBlog/public/archives/2021/page/2/index.html","20ba0316520d2f60499804e5a5bd990a"],["E:/GitHubBlog/public/archives/2021/page/3/index.html","be7244c3036547048a33aeafb63b4d00"],["E:/GitHubBlog/public/archives/index.html","4c9ce99ab6d5c73319307bc7a5c92e51"],["E:/GitHubBlog/public/archives/page/2/index.html","80097c00cfa464e7bc61f0f22e1dd85a"],["E:/GitHubBlog/public/archives/page/3/index.html","cfad1020f3c6366d68a29c0805728cec"],["E:/GitHubBlog/public/archives/page/4/index.html","5b0bacb3bed1345dcc23f3b3a68645b3"],["E:/GitHubBlog/public/archives/page/5/index.html","5ad0e698ce21e2a7228c2d32f6990952"],["E:/GitHubBlog/public/archives/page/6/index.html","a65e8b3870a0a478c9c91bd17da1fa9e"],["E:/GitHubBlog/public/archives/page/7/index.html","ccd50b567f3568a481049c1d8412e71e"],["E:/GitHubBlog/public/archives/page/8/index.html","e4e967ada0dee924ab59c43ab89c3632"],["E:/GitHubBlog/public/assets/css/APlayer.min.css","fbe994054426fadb2dff69d824c5c67a"],["E:/GitHubBlog/public/assets/js/APlayer.min.js","8f1017e7a73737e631ff95fa51e4e7d7"],["E:/GitHubBlog/public/assets/js/Meting.min.js","bfac0368480fd344282ec018d28f173d"],["E:/GitHubBlog/public/css/404.css","b1bb50e1fabe41adcec483f6427fb3aa"],["E:/GitHubBlog/public/css/index.css","2389364df340948a5b822422956c2c87"],["E:/GitHubBlog/public/css/var.css","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/img/algolia.svg","fd40b88ac5370a5353a50b8175c1f367"],["E:/GitHubBlog/public/img/avatar.png","6cc4a809d23e3d8946a299ae4ce4e4cd"],["E:/GitHubBlog/public/index.html","a380f2206935f54cb53217b5f314fb10"],["E:/GitHubBlog/public/js/copy.js","45aae54bf2e43ac27ecc41eb5e0bacf7"],["E:/GitHubBlog/public/js/fancybox.js","f84d626654b9bbc05720952b3effe062"],["E:/GitHubBlog/public/js/fireworks.js","35933ce61c158ef9c33b5e089fe757ac"],["E:/GitHubBlog/public/js/head.js","347edd99f8e3921b45fa617e839d8182"],["E:/GitHubBlog/public/js/hexo-theme-melody.js","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/js/katex.js","d971ba8b8dee1120ef66744b89cfd2b1"],["E:/GitHubBlog/public/js/scroll.js","a12ad7e37b9e325f8da3d0523d3757c7"],["E:/GitHubBlog/public/js/search/algolia.js","53160985d32d6fd66d98aa0e05b081ac"],["E:/GitHubBlog/public/js/search/local-search.js","1565b508bd866ed6fbd724a3858af5d8"],["E:/GitHubBlog/public/js/sidebar.js","d24db780974e661198eeb2e45f20a28f"],["E:/GitHubBlog/public/js/third-party/anime.min.js","9b4bbe6deb700e1c3606eab732f5eea5"],["E:/GitHubBlog/public/js/third-party/canvas-ribbon.js","09c181544ddff1db701db02ac30453e0"],["E:/GitHubBlog/public/js/third-party/jquery.fancybox.min.js","3c9fa1c1199cd4f874d855ecb1641335"],["E:/GitHubBlog/public/js/third-party/jquery.min.js","c9f5aeeca3ad37bf2aa006139b935f0a"],["E:/GitHubBlog/public/js/third-party/reveal/head.min.js","aad121203010122e05f1766d92385214"],["E:/GitHubBlog/public/js/third-party/reveal/highlight.min.js","44594243bec43813a16371af8fe7e105"],["E:/GitHubBlog/public/js/third-party/reveal/markdown.min.js","7ec4cef5a7fe3f0bf0eb4dc6d7bca114"],["E:/GitHubBlog/public/js/third-party/reveal/marked.min.js","c2a88705e206d71dc21fdc4445349127"],["E:/GitHubBlog/public/js/third-party/reveal/math.min.js","0a278fee2e57c530ab07f7d2d9ea8d96"],["E:/GitHubBlog/public/js/third-party/reveal/notes.min.js","89a0dfae4d706f9c75b317f686c3aa14"],["E:/GitHubBlog/public/js/third-party/reveal/reveal.min.js","8988419d67efb5fe93e291a357e26ec9"],["E:/GitHubBlog/public/js/third-party/reveal/zoom.min.js","9791f96e63e7d534cba2b67d4bda0419"],["E:/GitHubBlog/public/js/third-party/velocity.min.js","64da069aba987ea0512cf610600a56d1"],["E:/GitHubBlog/public/js/third-party/velocity.ui.min.js","c8ca438424a080620f7b2f4ee4b0fff1"],["E:/GitHubBlog/public/js/transition.js","911db4268f0f6621073afcced9e1bfef"],["E:/GitHubBlog/public/js/utils.js","3ff3423d966a1c351e9867813b3f6d36"],["E:/GitHubBlog/public/page/2/index.html","59621b0862980129d874bd09d79129c6"],["E:/GitHubBlog/public/page/3/index.html","892e929283fdc791494c4d677d7e80e8"],["E:/GitHubBlog/public/page/4/index.html","c57064563e975fe631a4726ddd7002de"],["E:/GitHubBlog/public/page/5/index.html","56a2dbe86943642cb4f784f6165e12f5"],["E:/GitHubBlog/public/page/6/index.html","95d8bd81c6c93e4385e68e8a5a86864a"],["E:/GitHubBlog/public/page/7/index.html","8bfecd959c72b197aaa0d1ae0c40b4c5"],["E:/GitHubBlog/public/page/8/index.html","5d785e14abb1a059435b628159905e2c"],["E:/GitHubBlog/public/tags/Android/index.html","380e0b8f0567e19c097c9dcca311a105"],["E:/GitHubBlog/public/tags/NLP/index.html","d2fbc7c5b83ee9a2a0008909b3274c2e"],["E:/GitHubBlog/public/tags/NLP/page/2/index.html","2c7cebe10d23a326f3b2a525db61b567"],["E:/GitHubBlog/public/tags/NLP/page/3/index.html","3c5a05d6409b168c1147e1504f18add0"],["E:/GitHubBlog/public/tags/R/index.html","0b6cbd31f75568dc64c125c4cffd6685"],["E:/GitHubBlog/public/tags/index.html","62886a9544051038c70a72ad61300583"],["E:/GitHubBlog/public/tags/java/index.html","4df1c79c94520abd0dc0b8cf5324f60f"],["E:/GitHubBlog/public/tags/java/page/2/index.html","d380c2d339432ef120bf1130ee5c20b2"],["E:/GitHubBlog/public/tags/leetcode/index.html","7e4b4666c482a1d8a336eb246aac3c68"],["E:/GitHubBlog/public/tags/python/index.html","9ccccc54e5a34631f19f2935ff3ed75d"],["E:/GitHubBlog/public/tags/pytorch/index.html","e828b253d41bc4c07114b92d0be14e60"],["E:/GitHubBlog/public/tags/代码/index.html","7818dc5c868cb5b1f52c9cdd6bda2b96"],["E:/GitHubBlog/public/tags/优化方法/index.html","405249bab1c2ebe1f9ce3ef3cdf6fe61"],["E:/GitHubBlog/public/tags/总结/index.html","96235dcdc1dd5816618511bbe1986d25"],["E:/GitHubBlog/public/tags/总结/page/2/index.html","4669a9ce7541432dfb702d7c9927bb0a"],["E:/GitHubBlog/public/tags/数据分析/index.html","b965caf3c5210c3f4243d38fd66c1042"],["E:/GitHubBlog/public/tags/数据挖掘/index.html","f195297285d56b3202340b8548a7435a"],["E:/GitHubBlog/public/tags/数据结构/index.html","20d79274a554509a244e5fd4edd9dbd1"],["E:/GitHubBlog/public/tags/机器学习/index.html","c3379f65d19f7d994743d852df2b677a"],["E:/GitHubBlog/public/tags/深度学习/index.html","b926728e74a2d82e87c3582c11b4fadb"],["E:/GitHubBlog/public/tags/爬虫/index.html","a6fbabed27011a77e3d988305030e15c"],["E:/GitHubBlog/public/tags/笔记/index.html","1b7c137d481ab3be444f0903118ba9e2"],["E:/GitHubBlog/public/tags/论文/index.html","70ffc9ce2c669152e0464944b826441e"],["E:/GitHubBlog/public/tags/论文/page/2/index.html","4df6c23db6de1bef2067a850a8886f33"],["E:/GitHubBlog/public/tags/论文/page/3/index.html","a08379785f7b3a69cf8e6beb156fd604"],["E:/GitHubBlog/public/tags/读书笔记/index.html","c9e4831673c2fcc2f38ef1c85b18c571"],["E:/GitHubBlog/public/tags/量化交易/index.html","64d2574ef826dce755913b6349d9f033"]];
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







