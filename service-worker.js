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

var precacheConfig = [["E:/GitHubBlog/public/2020/01/28/hello-world/index.html","324b13a24ebc38b3810e96d77fc4036e"],["E:/GitHubBlog/public/2020/02/08/python爬虫学习总结/index.html","973c742334c982a8034a9592533c2b3c"],["E:/GitHubBlog/public/2020/02/19/数据结构第一周-绪论/index.html","a1ca9d3c850854b6937a96bd0873f687"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-1/index.html","e47153b425bf33430f64fe66dae2c01c"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-2-创建数据集/index.html","0aee7f33922b6bf446d9b61ac81a1cb8"],["E:/GitHubBlog/public/2020/03/01/《R语言实战》学习总结-3-图形初阶/index.html","8801fef7674b206ee62edb613a395d13"],["E:/GitHubBlog/public/2020/03/02/数据结构第二周-线性结构1/index.html","782181caf1a786a43b9b79ab79d9242c"],["E:/GitHubBlog/public/2020/03/03/《R语言实战》学习总结-4-基本数据管理/index.html","885261740bd354bd207d57ae5fe7e037"],["E:/GitHubBlog/public/2020/03/14/数据结构第三周-线性结构2/index.html","99e7b61044cc780e4d81efb4b5d5caea"],["E:/GitHubBlog/public/2020/03/14/数据结构第四周-栈与队列/index.html","6c72639b51da5e2630b9c744c511096a"],["E:/GitHubBlog/public/2020/03/17/leetcode记录-1、88、20/index.html","40d74bf1fa03b6173f8aeade4059fef2"],["E:/GitHubBlog/public/2020/03/20/温度转换程序/index.html","3c07f70237e7325a89cc350af5ee0a75"],["E:/GitHubBlog/public/2020/03/22/数据结构第五周-递归/index.html","79112765497e51a37fdbb4e0a58ef326"],["E:/GitHubBlog/public/2020/03/22/数据结构第六周-串与数组/index.html","8d3aadaa04d96298a8828be6284c093d"],["E:/GitHubBlog/public/2020/04/02/leetcode记录-52/index.html","56dcde4ca3043814240a097c7b3801a2"],["E:/GitHubBlog/public/2020/04/17/Android第八周作业-每天第一次打开应用更新数据/index.html","e4eead256b9a1c682e5be2f172b89dd1"],["E:/GitHubBlog/public/2020/05/21/attention模型/index.html","67dc850e0fbae754b9a79ace78c5d16a"],["E:/GitHubBlog/public/2020/05/21/编码器与解码器框架/index.html","255e9a3cbad11bbf1fbc689d263bdfcf"],["E:/GitHubBlog/public/2020/07/11/思考，快与慢/index.html","1d7f3f37f6c522aa4eb339807331d229"],["E:/GitHubBlog/public/2020/07/13/20200713-0717总结/index.html","4931c68b26debf311cbd9a48a1088f43"],["E:/GitHubBlog/public/2020/07/13/ZeroShotCeres Zero-Shot Relation Extraction/index.html","e062cede546bf905836556cf089b28a5"],["E:/GitHubBlog/public/2020/07/20/Keyphrase Generation A Text Summarization Struggle/index.html","bce1498131b489fd8e62016f375e531c"],["E:/GitHubBlog/public/2020/07/20/TPR/index.html","99aaa940f0fca33957027fdb12b5d127"],["E:/GitHubBlog/public/2020/07/23/Keyphrase Generation for Scientific Document Retrieval/index.html","4312f1d64e2d7a1fbbe2b3abc01757c1"],["E:/GitHubBlog/public/2020/07/24/20200718-0724总结/index.html","0bc9b25e81d7f2d575bea67359cadb32"],["E:/GitHubBlog/public/2020/07/31/20200725-0731总结/index.html","9f1de4a17cb153c1dcd9f0e7a0c022f9"],["E:/GitHubBlog/public/2020/08/07/20200801-0807总结/index.html","365f81e5c3fa1a3e26bb98e28c4626ae"],["E:/GitHubBlog/public/2020/08/13/李宏毅-机器学习21/index.html","46573eb448d5a874cf8e2087d445fe96"],["E:/GitHubBlog/public/2020/08/14/20200808-0815总结/index.html","06907aef58f75da95873f896dce65720"],["E:/GitHubBlog/public/2020/09/03/20200816-0903总结/index.html","3c2eff16440ba7f0fc25eb6a1d1b66f6"],["E:/GitHubBlog/public/2020/09/16/20200914-0918总结/index.html","c751642b42f7cb91fcd9a4449401715e"],["E:/GitHubBlog/public/2020/09/25/20200919-0925总结/index.html","2b75348a4e62e16d142b21ae52f1180a"],["E:/GitHubBlog/public/2020/10/06/《Deep Keyphrase Generation》详细解析！/index.html","e35a925c138a120be058390f3bd05f3c"],["E:/GitHubBlog/public/2020/10/16/20201008-1017总结/index.html","4a0db45b654d2e89b9549d8178fb5ad3"],["E:/GitHubBlog/public/2020/10/20/特征选择的几种方法(一)————Filter过滤法/index.html","1256212308e0ff12a1a57971c6941a86"],["E:/GitHubBlog/public/2020/10/21/PCA/index.html","e761ae620d24a6a75955c87c7979aeac"],["E:/GitHubBlog/public/2020/10/22/20201018-1023总结/index.html","4f9da2a76937f12a8bb285c481427b14"],["E:/GitHubBlog/public/2020/10/24/20201024-1030/index.html","2e97fefe2df56126e93a8259d416270e"],["E:/GitHubBlog/public/2020/10/30/1030讨论记录/index.html","bd912f98a5bb9b938240c4d78a2dad81"],["E:/GitHubBlog/public/2020/11/03/20201103-总结/index.html","a67a2edfdad76ec8345eecfbaa9ad52c"],["E:/GitHubBlog/public/2020/11/05/线性回归与逻辑回归/index.html","5be45ffe73b26f2123d096ffb9263daf"],["E:/GitHubBlog/public/2020/11/07/决策树/index.html","a43633748e213f19b1421ca2a5387cb9"],["E:/GitHubBlog/public/2020/11/13/20201107-1113总结/index.html","dac4dcf619e55f76834b5d5e7e40f24a"],["E:/GitHubBlog/public/2020/11/15/20201114-119总结/index.html","e267ea9f425ee96d798ca8d4d490722e"],["E:/GitHubBlog/public/2020/11/16/手推一个SVM/index.html","7d5f0317b29a8ec6c139af81ab5fa190"],["E:/GitHubBlog/public/2020/11/27/20201127左右的总结/index.html","4d7b10207260f850cf8b6c602974275e"],["E:/GitHubBlog/public/2020/12/23/第一个深度学习模型（pytorch）/index.html","3fa2c638809c82009921f7233db25778"],["E:/GitHubBlog/public/2021/01/02/鲍鱼最终版/index.html","b7069707de5119f644efffcec356e71f"],["E:/GitHubBlog/public/2021/01/08/GSA_NumericalExperiment/index.html","520c2a8fe33547289959c0262a44cf1e"],["E:/GitHubBlog/public/2021/01/16/20210115-01总结/index.html","870b5748a5071ccaac0182bfa060e1de"],["E:/GitHubBlog/public/2021/01/17/20200117-0118总结——OpenNMT阅读（二）/index.html","d09001ace0fd308a023bafbe9b4ee0ad"],["E:/GitHubBlog/public/2021/01/17/OpenNMT源码解析/index.html","f7c152f99d7f4924b88f2880edca8c4e"],["E:/GitHubBlog/public/2021/01/19/重学数据结构——线性表/index.html","96ce532270c7b0a79f8e3758e89bf417"],["E:/GitHubBlog/public/2021/01/22/《深度学习入门-基于python的理论与实现》——第二章 感知机/index.html","a6fe49f74dd42cfd3c49de1fe49e953b"],["E:/GitHubBlog/public/2021/01/26/简单链表题/index.html","9749ccdd206a945d9e52d2f2b689182b"],["E:/GitHubBlog/public/2021/01/28/DeepKeyphraseGeneration/index.html","85a6ea03081ad5e0b2b497fdda5593c4"],["E:/GitHubBlog/public/2021/02/01/title模型详解/index.html","aaf77f057412d7a8480e2e25e07151ca"],["E:/GitHubBlog/public/2021/02/14/中等链表题/index.html","6edcc21c2e109f75312716e1509d0344"],["E:/GitHubBlog/public/2021/02/21/Teacher forcing in RNN/index.html","b347378d99a0e580e8bc5f9bf3691cc1"],["E:/GitHubBlog/public/2021/03/11/CopyNet代码消化记录/index.html","6eb6648dc1aabed353a46742540b42cb"],["E:/GitHubBlog/public/2021/03/11/王道机试指南习题/index.html","2b3715c743fcda563fa94f7426d0860b"],["E:/GitHubBlog/public/2021/03/19/AcWing代码记录/index.html","a04bf1726fa61a92dea8712bd2e7e88c"],["E:/GitHubBlog/public/2021/03/20/2021-03-26-week3组会/index.html","6bb01c69851cd63246a049288fec39a9"],["E:/GitHubBlog/public/2021/03/22/NEURAL MACHINE TRANSLATION模型/index.html","d45cc4b88214aa350b62e922f11bb06b"],["E:/GitHubBlog/public/2021/03/23/西瓜书学习——第一章 绪论/index.html","51e3da34692bb0d926676981ed023618"],["E:/GitHubBlog/public/2021/03/24/西瓜书南瓜书 第二章/index.html","1f256768e883bb522b580a16bbb6766a"],["E:/GitHubBlog/public/2021/04/04/python爬虫爬取新浪财经个股新闻研报/index.html","fd2bf16fb7f95464a9470c2613b9a434"],["E:/GitHubBlog/public/2021/04/04/西瓜书南瓜书 第三章/index.html","07addc363a6f0e1bf2891e65d63b75d1"],["E:/GitHubBlog/public/2021/04/09/ip被封了怎么办/index.html","4e98eb956584bf1eed6fac412ed66602"],["E:/GitHubBlog/public/2021/04/12/数据分析（机器学习）作业/index.html","76b37c0696d8a1af62fedd2af101c654"],["E:/GitHubBlog/public/2021/04/15/量化文本的方法/index.html","648fd18fd26d1c63b0b0d05019761f86"],["E:/GitHubBlog/public/2021/04/17/西瓜书南瓜书 第六章/index.html","50b037f6ae8835a1af93062ecbecf36f"],["E:/GitHubBlog/public/2021/04/18/scrapy新浪财经的重新爬取/index.html","115d5d2cd1aadb239d255630af9d9f6f"],["E:/GitHubBlog/public/2021/04/19/AcWing代码记录2/index.html","f7906091877bd3c779d240e1835591e0"],["E:/GitHubBlog/public/2021/04/20/基于情感词典的中文情感分析初探/index.html","25b4c8b83436e2f8fe3df29c3a65e8f6"],["E:/GitHubBlog/public/2021/04/21/一些量化文本的方法实现/index.html","534e6c1ced1a4f02b0eab9cf0073f47d"],["E:/GitHubBlog/public/2021/04/30/正则表达式笔记/index.html","bad066a2df7517b537e594b2f3692854"],["E:/GitHubBlog/public/2021/05/02/同花顺量化交易平台探索/index.html","5f956e5a459c99c2971fd152922e680f"],["E:/GitHubBlog/public/2021/05/03/C++语法基础课题目打卡/index.html","4dde5299b510e3b2f0d11e308ccd69d4"],["E:/GitHubBlog/public/2021/05/15/量化交易论文总结/index.html","22ed78d7432b4beb56ac66f2ad2ce830"],["E:/GitHubBlog/public/2021/05/19/AcWing代码记录3/index.html","31315bc199db88ae94817e0368eedf52"],["E:/GitHubBlog/public/2021/05/23/BFS和DFS题/index.html","7bd7c9bf6789759f81dd565afee5af89"],["E:/GitHubBlog/public/2021/06/08/论文笔记/index.html","c834ef8b55d31f4a2b40ab4dcd7b1496"],["E:/GitHubBlog/public/2021/06/14/Cross-Lingual Unsupervised Sentiment Classification with/index.html","64db6e3abf5066b7271902e1a9f03a3c"],["E:/GitHubBlog/public/2021/06/29/Pay Attention to MLPs阅读报告/index.html","2cf018ad09f8fe1e0b9f2885c7aa63d6"],["E:/GitHubBlog/public/archives/2020/01/index.html","3cb578f81740f88ab030f80abbc8553e"],["E:/GitHubBlog/public/archives/2020/02/index.html","983875a9e840c98cff569407dc982a03"],["E:/GitHubBlog/public/archives/2020/03/index.html","9cf42b3caf7292c7a20005d2b4536521"],["E:/GitHubBlog/public/archives/2020/04/index.html","272b90cc22f24c4d7032d2ed02587244"],["E:/GitHubBlog/public/archives/2020/05/index.html","88f63ab80900f4be1231417f02c1422b"],["E:/GitHubBlog/public/archives/2020/07/index.html","abc8a3e77c01084412919faf8f28679b"],["E:/GitHubBlog/public/archives/2020/08/index.html","f1780ced2ae82bb8ecd51e5f936c668c"],["E:/GitHubBlog/public/archives/2020/09/index.html","dbe0f8fefde09ed4d2447981c35dea03"],["E:/GitHubBlog/public/archives/2020/10/index.html","a864c8a5dc83600d67f3d34a97d6786f"],["E:/GitHubBlog/public/archives/2020/11/index.html","807f5303fd2949c9cd8f9fa0a8c251fa"],["E:/GitHubBlog/public/archives/2020/12/index.html","6021b1ea628a5934dd92cd01b6ee230c"],["E:/GitHubBlog/public/archives/2020/index.html","7bc00d7b46a54be4cc8edb288cacc86e"],["E:/GitHubBlog/public/archives/2020/page/2/index.html","0cc4795fab759e3d603a224d19bba4cf"],["E:/GitHubBlog/public/archives/2020/page/3/index.html","3075286c19fcb9961b0529fdcf60b65a"],["E:/GitHubBlog/public/archives/2020/page/4/index.html","09cf4bdbc301e1d7daf57936c5e29958"],["E:/GitHubBlog/public/archives/2020/page/5/index.html","1520853bc0c69113c4b588896a76d989"],["E:/GitHubBlog/public/archives/2021/01/index.html","be197b2ad166d51c66e2e83be7c997d0"],["E:/GitHubBlog/public/archives/2021/02/index.html","d85f4ad6d962501e55bceeacbc880723"],["E:/GitHubBlog/public/archives/2021/03/index.html","81f35ec3f67d9b925f507a1a5ad404c6"],["E:/GitHubBlog/public/archives/2021/04/index.html","e68eaf625637f29b343c3f24f46b4099"],["E:/GitHubBlog/public/archives/2021/04/page/2/index.html","baa5ce7953646fdeaeec9bee1fd46bb5"],["E:/GitHubBlog/public/archives/2021/05/index.html","d274a6c442c01ca1d66352e903b1cc82"],["E:/GitHubBlog/public/archives/2021/06/index.html","74a88227a14e317e59c384871dacb4f9"],["E:/GitHubBlog/public/archives/2021/index.html","8e8d442db1bd8e2148a45ca840965499"],["E:/GitHubBlog/public/archives/2021/page/2/index.html","d1d66929a482694401e9e77c1475ea22"],["E:/GitHubBlog/public/archives/2021/page/3/index.html","94256d05be058a45d0cb36515e5bfc98"],["E:/GitHubBlog/public/archives/2021/page/4/index.html","29d18fcde84343fb2214be943d99fe8a"],["E:/GitHubBlog/public/archives/index.html","469a21a946267c2477d349649e6b8f5c"],["E:/GitHubBlog/public/archives/page/2/index.html","4de02ac3754ac2197713e53fbdc53f61"],["E:/GitHubBlog/public/archives/page/3/index.html","47ca17f5260b52ab76b0f5c4d16e5679"],["E:/GitHubBlog/public/archives/page/4/index.html","9bbfaa109fdd52a77fe5c965e304fb59"],["E:/GitHubBlog/public/archives/page/5/index.html","68507fc1ea6fde4f0d47f34449e7ae0d"],["E:/GitHubBlog/public/archives/page/6/index.html","8486bad0c7025efd0352a43710fd36d8"],["E:/GitHubBlog/public/archives/page/7/index.html","96f8220c1f08067f80172676642f6147"],["E:/GitHubBlog/public/archives/page/8/index.html","01086eb1cc90abd5e4bf702230b003a0"],["E:/GitHubBlog/public/archives/page/9/index.html","f6496770b4ec4397ac3bf8cdc71bb402"],["E:/GitHubBlog/public/assets/css/APlayer.min.css","fbe994054426fadb2dff69d824c5c67a"],["E:/GitHubBlog/public/assets/js/APlayer.min.js","8f1017e7a73737e631ff95fa51e4e7d7"],["E:/GitHubBlog/public/assets/js/Meting.min.js","bfac0368480fd344282ec018d28f173d"],["E:/GitHubBlog/public/categories/书籍学习/index.html","5ec5903208bbafa82325d073c133c476"],["E:/GitHubBlog/public/css/font.css","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/css/index.css","c86084ab63c8ac8d43ecc89e8c98ae41"],["E:/GitHubBlog/public/css/post.css","d746e759ed361785cd87827baa1a6ef1"],["E:/GitHubBlog/public/css/reset.css","4fd030817d0e23aa84743dabb65009a0"],["E:/GitHubBlog/public/css/tocbot.css","e8f0173e7c5216e5359587a88a570b77"],["E:/GitHubBlog/public/css/var.css","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/img/icon-left-arrow.svg","6aa3f4fd9ad7a807fb4f7ef626dfd0c3"],["E:/GitHubBlog/public/img/icon-menu-sprite.svg","8f7f2745eb92cf88761f378b0ff98ebe"],["E:/GitHubBlog/public/index.html","e75c0d76a4d4b9165038b82d76cc5f08"],["E:/GitHubBlog/public/js/paper.js","551d4d268064033a6b0625768834ec3e"],["E:/GitHubBlog/public/js/tocbot.js","427555b1fdee580e22f144e233498068"],["E:/GitHubBlog/public/page/2/index.html","8e17d9a39e51b7dc45e33b309de32921"],["E:/GitHubBlog/public/page/3/index.html","7f2b3d0ee4ba5961158c0884303d7281"],["E:/GitHubBlog/public/page/4/index.html","aa1ddf8e2fd975fbbf088c132276c5db"],["E:/GitHubBlog/public/page/5/index.html","0eca1d419c397b383d8d2538c5ad9acc"],["E:/GitHubBlog/public/page/6/index.html","efc60c180864d1736f21127a1ffab67b"],["E:/GitHubBlog/public/page/7/index.html","6343791f72aea2a9bf56340ec5430b49"],["E:/GitHubBlog/public/page/8/index.html","01a7bbdaf17cb8f65185391229ee6f1b"],["E:/GitHubBlog/public/page/9/index.html","a8b1e6fc11517f91d9400125bbfd3801"],["E:/GitHubBlog/public/tags/Android/index.html","db5a3e4af4e75eaa58cf9fe5b36a1d97"],["E:/GitHubBlog/public/tags/NLP/index.html","16d3b620de02da23617eb44c8045b465"],["E:/GitHubBlog/public/tags/NLP/page/2/index.html","47ca84f32e06278dce98d1824e5607a0"],["E:/GitHubBlog/public/tags/NLP/page/3/index.html","c7af74ce223fd710b2f5506a1cea99b4"],["E:/GitHubBlog/public/tags/R/index.html","778e29e810437adce43698244b321f40"],["E:/GitHubBlog/public/tags/index.html","535c2918b137de53f067423d0a694f73"],["E:/GitHubBlog/public/tags/java/index.html","dba04b15a1a93d8b52792bb63f25b9c3"],["E:/GitHubBlog/public/tags/java/page/2/index.html","9ee8d84ababdd5c2cfd867f7abcdff71"],["E:/GitHubBlog/public/tags/kpg/index.html","ea0d45a7d2c5e009f5a0bdbc223c36b0"],["E:/GitHubBlog/public/tags/leetcode/index.html","dcdeb9ce61fc4c7f6b55ab806ae7865b"],["E:/GitHubBlog/public/tags/python/index.html","64ffb361e218529ea02ce2d0f6ef8dde"],["E:/GitHubBlog/public/tags/pytorch/index.html","f672a5f4318a666707ca3d1b43320815"],["E:/GitHubBlog/public/tags/《南瓜书》/index.html","5570fa1f1f859dce97a61ea2cc8713f3"],["E:/GitHubBlog/public/tags/《正则表达式必知必会》/index.html","af20009ac07f853c24fd1ba065931b7a"],["E:/GitHubBlog/public/tags/《西瓜书》/index.html","477fc8888748e65289242ffbf5d9fc37"],["E:/GitHubBlog/public/tags/优化方法/index.html","7cd4882e15f25ad5ceb9b51593c9be95"],["E:/GitHubBlog/public/tags/复制机制/index.html","ffadcbc6d82861e0ebf81b9e531aa46d"],["E:/GitHubBlog/public/tags/总结/index.html","2ea0bf8ca01db7584efdbc39c8675366"],["E:/GitHubBlog/public/tags/总结/page/2/index.html","a3ec1171fe83af29c1dd3fba6383f0fa"],["E:/GitHubBlog/public/tags/推荐系统/index.html","28514f024170abac33a8b3e8faba8fe0"],["E:/GitHubBlog/public/tags/数据分析/index.html","e4bd21147086a4a11a44241b0e61e9d6"],["E:/GitHubBlog/public/tags/数据挖掘/index.html","3651fd29bf437ce0cc6b77b4a74747dc"],["E:/GitHubBlog/public/tags/数据科学实战/index.html","7ae442524342feb30c8183d0f260416b"],["E:/GitHubBlog/public/tags/数据结构/index.html","24e727fbe856a607efe231485b77fb5d"],["E:/GitHubBlog/public/tags/机器学习/index.html","f475bb6b0b55330dfb89ae14da6c993f"],["E:/GitHubBlog/public/tags/机试准备/index.html","31d6a16b2d63484b9177027c17d5b501"],["E:/GitHubBlog/public/tags/深度学习/index.html","b1d7c00e7c586d0f33156cca5ef951af"],["E:/GitHubBlog/public/tags/爬虫/index.html","4e28eb13ee5e2ceb0aaee61dc6ce2a5c"],["E:/GitHubBlog/public/tags/笔记/index.html","f77bbdde0f9a6edd9ce2f2c7348dfcfc"],["E:/GitHubBlog/public/tags/算法/index.html","94a900d946932202fe27cea8c248a3c5"],["E:/GitHubBlog/public/tags/论文/index.html","2f9a8723cda951fdb7fded1a3bea8327"],["E:/GitHubBlog/public/tags/论文/page/2/index.html","5f26ff1c4767f5340adef5af880a1ea6"],["E:/GitHubBlog/public/tags/论文/page/3/index.html","aaee8176805acb7a404d1eadae5f4a8c"],["E:/GitHubBlog/public/tags/读书笔记/index.html","422714869739b9b76ea2c9b5683a52b6"],["E:/GitHubBlog/public/tags/量化交易/index.html","b5011cb8dd1f1e2765cb4a8e696d8c44"],["E:/GitHubBlog/public/tags/项目-量化交易/index.html","6ca88e905c7c38df59db648bb425d21b"]];
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







