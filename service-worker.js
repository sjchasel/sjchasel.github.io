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

var precacheConfig = [["E:/GitHubBlog/public/2020/01/28/hello-world/index.html","0eed3ecda156eb6513002a6c686bf69b"],["E:/GitHubBlog/public/2020/02/08/python爬虫学习总结/index.html","0af377cc252ee4b4cc58a727815f4c50"],["E:/GitHubBlog/public/2020/02/19/数据结构第一周-绪论/index.html","47a746ada8ba6a892675d7ecf96ee385"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-1/index.html","fb2a9c25c8c7faeac5fb76086a9233ca"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-2-创建数据集/index.html","e65b077ee76f3e1135a7737e955fa07a"],["E:/GitHubBlog/public/2020/03/01/《R语言实战》学习总结-3-图形初阶/index.html","18343207702df5752a45d2b1fd1e536b"],["E:/GitHubBlog/public/2020/03/02/数据结构第二周-线性结构1/index.html","6a4ee198605f6b65504478ab0c025623"],["E:/GitHubBlog/public/2020/03/03/《R语言实战》学习总结-4-基本数据管理/index.html","53f24787801548898fa50b5f3c89ec4a"],["E:/GitHubBlog/public/2020/03/14/数据结构第三周-线性结构2/index.html","d9654e9dd5158a2de601967f64db5a29"],["E:/GitHubBlog/public/2020/03/14/数据结构第四周-栈与队列/index.html","07318bc11bc85e8d0345335c725758ac"],["E:/GitHubBlog/public/2020/03/17/leetcode记录-1、88、20/index.html","821ffe6ea996cd7b1ecfb8b64a6a506a"],["E:/GitHubBlog/public/2020/03/20/温度转换程序/index.html","79f37f0bc66b25cabc184eada3800ce9"],["E:/GitHubBlog/public/2020/03/22/数据结构第五周-递归/index.html","e262a98114a47a3b6284e8434c4856e3"],["E:/GitHubBlog/public/2020/03/22/数据结构第六周-串与数组/index.html","cbb4af9782c66c369551672e26442358"],["E:/GitHubBlog/public/2020/04/02/leetcode记录-52/index.html","156c832bd60903919f1cc3fe8a9c5754"],["E:/GitHubBlog/public/2020/04/17/Android第八周作业-每天第一次打开应用更新数据/index.html","0dd38ee7d38ec2b5d4427ecee081d181"],["E:/GitHubBlog/public/2020/05/21/attention模型/index.html","f4bbb8ae1dceabc2084de098213a2cc6"],["E:/GitHubBlog/public/2020/05/21/编码器与解码器框架/index.html","fc4544691e93f2f9784dae7871e9bcbe"],["E:/GitHubBlog/public/2020/07/11/思考，快与慢/index.html","c962d2af746c2ec6a5b122b5922616da"],["E:/GitHubBlog/public/2020/07/13/20200713-0717总结/index.html","2ab09159c996ccc8bcaa5d6d206cdadf"],["E:/GitHubBlog/public/2020/07/13/ZeroShotCeres Zero-Shot Relation Extraction/index.html","bc38ad8a5a6491530a36a9c4aeebfd72"],["E:/GitHubBlog/public/2020/07/20/Keyphrase Generation A Text Summarization Struggle/index.html","2db92442c788c14d1882f7f4b965cef6"],["E:/GitHubBlog/public/2020/07/20/TPR/index.html","d652af5854719757d25f64dad9d3cfa6"],["E:/GitHubBlog/public/2020/07/23/Keyphrase Generation for Scientific Document Retrieval/index.html","75a634576ef85e26fa7da1836e13f8bb"],["E:/GitHubBlog/public/2020/07/24/20200718-0724总结/index.html","00a90295e49c508c0cf482269f864d2d"],["E:/GitHubBlog/public/2020/07/31/20200725-0731总结/index.html","45c93f07a78c0e66d1b04a151426935b"],["E:/GitHubBlog/public/2020/08/07/20200801-0807总结/index.html","0730195cedaef933fb031fc671a364e6"],["E:/GitHubBlog/public/2020/08/13/李宏毅-机器学习21/index.html","ab7456b3c0a1053cd02fea58c458983b"],["E:/GitHubBlog/public/2020/08/14/20200808-0815总结/index.html","435d423e02eb98f0c2cacaf1294c3859"],["E:/GitHubBlog/public/2020/09/03/20200816-0903总结/index.html","10edd6715c5560cd02adcfeaf1c1a7a0"],["E:/GitHubBlog/public/2020/09/16/20200914-0918总结/index.html","f090cd2de78f023bf1d2a774220b6efe"],["E:/GitHubBlog/public/2020/09/25/20200919-0925总结/index.html","e2dda4004bb8b4bfac8d25be9e4211a4"],["E:/GitHubBlog/public/2020/10/06/《Deep Keyphrase Generation》详细解析！/index.html","3a62bd78757bd0827f8b02052baf98d0"],["E:/GitHubBlog/public/2020/10/16/20201008-1017总结/index.html","4cbe576b22b5e6aac114bda1b1b974ea"],["E:/GitHubBlog/public/2020/10/20/特征选择的几种方法(一)————Filter过滤法/index.html","6ed3dd8162639984f2c6346a1d8b22de"],["E:/GitHubBlog/public/2020/10/21/PCA/index.html","580e33be5f8b3a2a503878977e481270"],["E:/GitHubBlog/public/2020/10/22/20201018-1023总结/index.html","d33e74b36f7fb0f7556da044ce44a48a"],["E:/GitHubBlog/public/2020/10/24/20201024-1030/index.html","835bc6a7b85660605b01ac739d24a318"],["E:/GitHubBlog/public/2020/10/30/1030讨论记录/index.html","26ae7b44498127537519989a1d330b10"],["E:/GitHubBlog/public/2020/11/03/20201103-总结/index.html","b9275afa9c4941147bbb3b3224ff73a4"],["E:/GitHubBlog/public/2020/11/05/线性回归与逻辑回归/index.html","f43282635b98bcab31fdae329e8c0247"],["E:/GitHubBlog/public/2020/11/07/决策树/index.html","8f654042e8348836c733a64e7e8b4a78"],["E:/GitHubBlog/public/2020/11/13/20201107-1113总结/index.html","0d75537df2d1a46df6d49007768813cf"],["E:/GitHubBlog/public/2020/11/15/20201114-119总结/index.html","710c485ee39b7cd092f9027d6f113ecf"],["E:/GitHubBlog/public/2020/11/16/手推一个SVM/index.html","728ae6006d3b7d8c17e5724ab70a4cc2"],["E:/GitHubBlog/public/2020/11/27/20201127左右的总结/index.html","07e9e75c3dfa41042f85abc01fa3682c"],["E:/GitHubBlog/public/2020/12/23/第一个深度学习模型（pytorch）/index.html","67513588ef780283be807d3910e5e98b"],["E:/GitHubBlog/public/2021/01/02/鲍鱼最终版/index.html","671405e354eaf6d3e6e23a8b5f7d08af"],["E:/GitHubBlog/public/2021/01/08/GSA_NumericalExperiment/index.html","5954454ed62df888a5df20ab320d297e"],["E:/GitHubBlog/public/2021/01/16/20210115-01总结/index.html","ab0b070ddcb71baafc0c39d6453895d8"],["E:/GitHubBlog/public/2021/01/17/20200117-0118总结——OpenNMT阅读（二）/index.html","9c4d9806a73db4ec30cfa6530525c82d"],["E:/GitHubBlog/public/2021/01/17/OpenNMT源码解析/index.html","80a4120e313257b47d1a0887dc5857d1"],["E:/GitHubBlog/public/2021/01/19/重学数据结构——线性表/index.html","ef989d85ed75d6c5b32e7eb5b62b6ccf"],["E:/GitHubBlog/public/2021/01/22/《深度学习入门-基于python的理论与实现》——第二章 感知机/index.html","07b9f17a4b03bc57fd50b96921977492"],["E:/GitHubBlog/public/2021/01/26/title复现中的问题/index.html","1a8d06fdd165bfdbed193a63fdd626ec"],["E:/GitHubBlog/public/2021/01/26/简单链表题/index.html","824f7756418b15d83aab492de13d677b"],["E:/GitHubBlog/public/2021/01/28/DeepKeyphraseGeneration/index.html","abe29a13268aec88b168896cbd135c2a"],["E:/GitHubBlog/public/2021/02/01/title模型详解/index.html","34e0114f077a897c8681b3a63baaa6c5"],["E:/GitHubBlog/public/2021/02/14/中等链表题/index.html","4af4589a784a11dd59f9a6491b733bd7"],["E:/GitHubBlog/public/2021/02/21/Teacher forcing in RNN/index.html","33bc65fe61bb2a82caa9673596539e0e"],["E:/GitHubBlog/public/2021/03/11/CopyNet代码消化记录/index.html","70af070c85f74f6bc254b2cd2e29dab5"],["E:/GitHubBlog/public/2021/03/11/王道机试指南习题/index.html","c6ae18faf087293f1bf90dd0dd9ccf36"],["E:/GitHubBlog/public/2021/03/13/野生kpg代码挣扎记录/index.html","16923486ee0b70be90fb3725ea7dd5fe"],["E:/GitHubBlog/public/2021/03/19/AcWing代码记录/index.html","1280c7eeac0e8b8aa8ffc9d68bfa6d48"],["E:/GitHubBlog/public/2021/03/20/2021-03-26-week3组会/index.html","2865ab2fb781f315590ba8362561a7d2"],["E:/GitHubBlog/public/2021/03/22/NEURAL MACHINE TRANSLATION模型/index.html","cb816bb040fe31092a4a158697c7a81a"],["E:/GitHubBlog/public/2021/03/23/西瓜书学习——第一章 绪论/index.html","e6134fd0ea17190e552aec8b8b8836e8"],["E:/GitHubBlog/public/2021/03/24/西瓜书南瓜书 第二章/index.html","e973a08e1fe94db1f7adc14352b3a8af"],["E:/GitHubBlog/public/2021/04/01/野生kpg代码挣扎记录2/index.html","55b21aec73a9aaed47545a686c07f43b"],["E:/GitHubBlog/public/2021/04/04/python爬虫爬取新浪财经个股新闻研报/index.html","04ff260b531faaff2b7c3fb82a06dafa"],["E:/GitHubBlog/public/2021/04/04/西瓜书南瓜书 第三章/index.html","d8ad919755339b1b4cc95f0b625fb1a3"],["E:/GitHubBlog/public/2021/04/09/ip被封了怎么办/index.html","fc346aa154ea2c07d84f66cb5bcd4e9f"],["E:/GitHubBlog/public/2021/04/12/修改代码流水账——数据预处理部分/index.html","1ccaf5486b3e14faf4e620a8652acb23"],["E:/GitHubBlog/public/2021/04/12/数据分析（机器学习）作业/index.html","9b7e8b626cd5affc457fbfa012a51601"],["E:/GitHubBlog/public/2021/04/15/量化文本的方法/index.html","3694323e23587060bcb902a963e38832"],["E:/GitHubBlog/public/archives/2020/01/index.html","25a95e856bfe2969a9a5a0b6b75271e6"],["E:/GitHubBlog/public/archives/2020/02/index.html","4d3542e955a0b61eebd0e7c84bb914ff"],["E:/GitHubBlog/public/archives/2020/03/index.html","85d08b7ac9b434db0beb617ed0f96fa7"],["E:/GitHubBlog/public/archives/2020/04/index.html","a992622e9b75be4384220f4a1cab211b"],["E:/GitHubBlog/public/archives/2020/05/index.html","2275263f9798ec596c61f579d664cd89"],["E:/GitHubBlog/public/archives/2020/07/index.html","f719ffa8df60ffe2ac21d105d635b151"],["E:/GitHubBlog/public/archives/2020/08/index.html","43cb2c116c70c1221b24ba4627f2fda2"],["E:/GitHubBlog/public/archives/2020/09/index.html","ccdd51bb1ceb0542540728ee6abb50fb"],["E:/GitHubBlog/public/archives/2020/10/index.html","e294ba6ae119f38ba91ec60ad56aff56"],["E:/GitHubBlog/public/archives/2020/11/index.html","c9cd7a531e743818752f55392883c7a3"],["E:/GitHubBlog/public/archives/2020/12/index.html","dc522f4b410f79426204fd375c22b12a"],["E:/GitHubBlog/public/archives/2020/index.html","451c2b3834b0b5488c0075c36e573df2"],["E:/GitHubBlog/public/archives/2020/page/2/index.html","487b6c562fc83fc5602b6a1fbc90cc32"],["E:/GitHubBlog/public/archives/2020/page/3/index.html","c0329f93fc08fada21a9f846dba43b99"],["E:/GitHubBlog/public/archives/2020/page/4/index.html","66fbeedac15662ea31ef8ace3da3d992"],["E:/GitHubBlog/public/archives/2020/page/5/index.html","59da69efb150e4c4b6054a770a711ef5"],["E:/GitHubBlog/public/archives/2021/01/index.html","fb2830591ba882c8732a36298c3d0fb2"],["E:/GitHubBlog/public/archives/2021/02/index.html","18c5378ec071397df5c92c2ea24ef2b2"],["E:/GitHubBlog/public/archives/2021/03/index.html","3c26d563cb05092474a485fe81642b64"],["E:/GitHubBlog/public/archives/2021/04/index.html","fedd09f49fb395d9f48dd09751a9b083"],["E:/GitHubBlog/public/archives/2021/index.html","ef85707cf1477f58480541613ed03f62"],["E:/GitHubBlog/public/archives/2021/page/2/index.html","cec9c21253d8a82093bde74562072681"],["E:/GitHubBlog/public/archives/2021/page/3/index.html","724e64f8a8154d2c6704f5469b6db556"],["E:/GitHubBlog/public/archives/index.html","e9670c784adff71d82db3e14e830db34"],["E:/GitHubBlog/public/archives/page/2/index.html","c1f5a8c4a52598883c54aa161ecb7d33"],["E:/GitHubBlog/public/archives/page/3/index.html","31fb9f0c517d581c8ae4524bae63d7f2"],["E:/GitHubBlog/public/archives/page/4/index.html","37c48063fa3359fcf13103b6b291b11b"],["E:/GitHubBlog/public/archives/page/5/index.html","e7c4d230a3708ac0e9d396e1f9713be5"],["E:/GitHubBlog/public/archives/page/6/index.html","deff1772cb667d215826e14c3adb8b0f"],["E:/GitHubBlog/public/archives/page/7/index.html","bec8e70f2b3683f36c4a88d8484d5d71"],["E:/GitHubBlog/public/archives/page/8/index.html","09a7c3ada470a55a6986d3587490b889"],["E:/GitHubBlog/public/assets/css/APlayer.min.css","fbe994054426fadb2dff69d824c5c67a"],["E:/GitHubBlog/public/assets/js/APlayer.min.js","8f1017e7a73737e631ff95fa51e4e7d7"],["E:/GitHubBlog/public/assets/js/Meting.min.js","bfac0368480fd344282ec018d28f173d"],["E:/GitHubBlog/public/css/font.css","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/css/index.css","c86084ab63c8ac8d43ecc89e8c98ae41"],["E:/GitHubBlog/public/css/post.css","d746e759ed361785cd87827baa1a6ef1"],["E:/GitHubBlog/public/css/reset.css","4fd030817d0e23aa84743dabb65009a0"],["E:/GitHubBlog/public/css/tocbot.css","e8f0173e7c5216e5359587a88a570b77"],["E:/GitHubBlog/public/css/var.css","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/img/icon-left-arrow.svg","6aa3f4fd9ad7a807fb4f7ef626dfd0c3"],["E:/GitHubBlog/public/img/icon-menu-sprite.svg","8f7f2745eb92cf88761f378b0ff98ebe"],["E:/GitHubBlog/public/index.html","1cd882c2d918ec14a23d4ecc57bcf6b4"],["E:/GitHubBlog/public/js/paper.js","551d4d268064033a6b0625768834ec3e"],["E:/GitHubBlog/public/js/tocbot.js","427555b1fdee580e22f144e233498068"],["E:/GitHubBlog/public/page/2/index.html","9d6ba8f26147f90eb2baec8f30cec35d"],["E:/GitHubBlog/public/page/3/index.html","e92d218e0edb1cdce7136fae07b55f0d"],["E:/GitHubBlog/public/page/4/index.html","f3a279c17cb647815b2d2a2c4ecca058"],["E:/GitHubBlog/public/page/5/index.html","aa2a94294f3bc4a7fdf1220bc0ca3b7a"],["E:/GitHubBlog/public/page/6/index.html","1832ca4b243382ba56941c2a2f24d444"],["E:/GitHubBlog/public/page/7/index.html","d51d51b71275b76308d929cbbf3a30a2"],["E:/GitHubBlog/public/page/8/index.html","82b993647920fe91e311d23f7d743e69"],["E:/GitHubBlog/public/tags/Android/index.html","36164d6abc918c346474ff507518813f"],["E:/GitHubBlog/public/tags/R/index.html","ba0a35e651678ef62979dfae02a5ea33"],["E:/GitHubBlog/public/tags/index.html","ef28c4c60a3a53aeab2ed9f5f6da3eac"],["E:/GitHubBlog/public/tags/java/index.html","d3914806acc0b4196d84f17ff9089970"],["E:/GitHubBlog/public/tags/java/page/2/index.html","ca570cb44bb732497fd57ee12322f4d3"],["E:/GitHubBlog/public/tags/kpg/index.html","bd41e90bdaa05b5a7667d4d6e93733df"],["E:/GitHubBlog/public/tags/leetcode/index.html","f69e478d33782259052e6323642bf107"],["E:/GitHubBlog/public/tags/nlp/index.html","a4a9a87e5083db8044312a451d8d05a6"],["E:/GitHubBlog/public/tags/nlp/page/2/index.html","792119abd1758aee05c7001a02a72bec"],["E:/GitHubBlog/public/tags/nlp/page/3/index.html","e5ab465e22797bc53d099dbd2e8c78bb"],["E:/GitHubBlog/public/tags/python/index.html","28b8336b57ffb6c2aac94b29f95a8bc0"],["E:/GitHubBlog/public/tags/pytorch/index.html","f9a70d147393a5a2e0bcdd2998bd6e28"],["E:/GitHubBlog/public/tags/《南瓜书》/index.html","f23dae8d37b0e4f16237be4cec215ab5"],["E:/GitHubBlog/public/tags/《西瓜书》/index.html","93c0b31db3fac1fa3333681cce2ba878"],["E:/GitHubBlog/public/tags/代码/index.html","ed8e9fb183eed7e7f3caf4ea7309a4be"],["E:/GitHubBlog/public/tags/优化方法/index.html","72d6f5517933807519d93db1f5068506"],["E:/GitHubBlog/public/tags/复制机制/index.html","ee15a40b783c0e103712610a6dbb307e"],["E:/GitHubBlog/public/tags/总结/index.html","e5efac864f8c4baaaac1b82115d8b36c"],["E:/GitHubBlog/public/tags/总结/page/2/index.html","c1ddf40e11b04f2b37119876441d5656"],["E:/GitHubBlog/public/tags/数据分析/index.html","b6fd091b0205884795989eca3a1f809e"],["E:/GitHubBlog/public/tags/数据挖掘/index.html","7b19a505f99865a04ec4f8ffb1289102"],["E:/GitHubBlog/public/tags/数据科学实战/index.html","cf389cd65ec7880351282b6f18f616fd"],["E:/GitHubBlog/public/tags/数据结构/index.html","2e258aee6de5ea50fa2ea1973a0b413e"],["E:/GitHubBlog/public/tags/机器学习/index.html","ea355299f93e115094152e680daa83a1"],["E:/GitHubBlog/public/tags/机试准备/index.html","3ff668478e737b6cdea7c13eb944c5f5"],["E:/GitHubBlog/public/tags/深度学习/index.html","f5ee7b8dfda3f66e74fbe15471256f14"],["E:/GitHubBlog/public/tags/爬虫/index.html","1a5924795df1d2a40634813430acb5f8"],["E:/GitHubBlog/public/tags/笔记/index.html","f22b9f78c5689a1a4798c2e8f7fd7ea5"],["E:/GitHubBlog/public/tags/算法/index.html","ba6e4fed506f95cd3f67083bcf58d859"],["E:/GitHubBlog/public/tags/论文/index.html","ffe549261d255e950923f2d170ea6d75"],["E:/GitHubBlog/public/tags/论文/page/2/index.html","0a147e08f41fa3e6e64b5d8dead40789"],["E:/GitHubBlog/public/tags/论文/page/3/index.html","5c22f2739f54cb108d959cf7f9430f21"],["E:/GitHubBlog/public/tags/读书笔记/index.html","d1da9bf9bede285483972203ab83ac1a"],["E:/GitHubBlog/public/tags/项目-量化交易/index.html","b9f974100ad5afd7ffe67f391e91a920"]];
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







