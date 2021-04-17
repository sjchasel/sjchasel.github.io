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

var precacheConfig = [["E:/GitHubBlog/public/2020/01/28/hello-world/index.html","7ed5a21265407bf684969da1b57b2b26"],["E:/GitHubBlog/public/2020/02/08/python爬虫学习总结/index.html","829f70418ac2795e9935ed5277b6f6e1"],["E:/GitHubBlog/public/2020/02/19/数据结构第一周-绪论/index.html","6a8f48ae71d4246a6947b0a5f0591e34"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-1/index.html","9eab7fbfbb82dee4b1a69403c91e7331"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-2-创建数据集/index.html","61f5b8eda40732fc4b428db5bd918c80"],["E:/GitHubBlog/public/2020/03/01/《R语言实战》学习总结-3-图形初阶/index.html","29b59db4e85d309f14c02b131df42e4e"],["E:/GitHubBlog/public/2020/03/02/数据结构第二周-线性结构1/index.html","d288ea3f6f69a47bc464b0e0c93f02a5"],["E:/GitHubBlog/public/2020/03/03/《R语言实战》学习总结-4-基本数据管理/index.html","c01bee4c43e9947d070d8da0890f591a"],["E:/GitHubBlog/public/2020/03/14/数据结构第三周-线性结构2/index.html","8e60b7d06997d430cf9b810b1d1bdc59"],["E:/GitHubBlog/public/2020/03/14/数据结构第四周-栈与队列/index.html","1e8b3f0ae4052f655c1f7c39cd7a2156"],["E:/GitHubBlog/public/2020/03/17/leetcode记录-1、88、20/index.html","ba55f0b3225d6b0a7c69f378c5ddc1a3"],["E:/GitHubBlog/public/2020/03/20/温度转换程序/index.html","c2f97b342691cc1dc09655ac5c4168b8"],["E:/GitHubBlog/public/2020/03/22/数据结构第五周-递归/index.html","b03d148f551bb8bc4078dfb8e7046a71"],["E:/GitHubBlog/public/2020/03/22/数据结构第六周-串与数组/index.html","8584817a32036caf980b7b7e6b507d97"],["E:/GitHubBlog/public/2020/04/02/leetcode记录-52/index.html","546f1541786f187dd1b78c5e9a63b9d0"],["E:/GitHubBlog/public/2020/04/17/Android第八周作业-每天第一次打开应用更新数据/index.html","ba787225df62e37b3ceaa38e5fdf1799"],["E:/GitHubBlog/public/2020/05/21/attention模型/index.html","bcf0b878ac3a49006bc52b67723ad88e"],["E:/GitHubBlog/public/2020/05/21/编码器与解码器框架/index.html","370a7dbc1589600f69870602ce63d436"],["E:/GitHubBlog/public/2020/07/11/思考，快与慢/index.html","764a0ebdb7f7feda1736aacb02f1a20c"],["E:/GitHubBlog/public/2020/07/13/20200713-0717总结/index.html","f77e7a28430de84c7382f37a15fcb279"],["E:/GitHubBlog/public/2020/07/13/ZeroShotCeres Zero-Shot Relation Extraction/index.html","f3e1bb9ef0c86396451868697a740dd3"],["E:/GitHubBlog/public/2020/07/20/Keyphrase Generation A Text Summarization Struggle/index.html","511d577b75a479eef2d6ed30eb14d94b"],["E:/GitHubBlog/public/2020/07/20/TPR/index.html","3e4dac70ab436be16cc14c57ee0e5165"],["E:/GitHubBlog/public/2020/07/23/Keyphrase Generation for Scientific Document Retrieval/index.html","62218c1800f8641a6dc9079f3e58e073"],["E:/GitHubBlog/public/2020/07/24/20200718-0724总结/index.html","29cfcf8bdfe8d9be5662e3d78cac21ad"],["E:/GitHubBlog/public/2020/07/31/20200725-0731总结/index.html","5f9222c16f4fc3566b00a06dbd28347e"],["E:/GitHubBlog/public/2020/08/07/20200801-0807总结/index.html","93b6b21125daa8b24c6dbd09596c3a47"],["E:/GitHubBlog/public/2020/08/13/李宏毅-机器学习21/index.html","18e14bd68dad2ab3161188e83d89a20e"],["E:/GitHubBlog/public/2020/08/14/20200808-0815总结/index.html","c16519e29a5feaa5fe5f51b119509104"],["E:/GitHubBlog/public/2020/09/03/20200816-0903总结/index.html","e20c072e5360a79c0525fb0013adad71"],["E:/GitHubBlog/public/2020/09/16/20200914-0918总结/index.html","6663344b837a9775cec0bd1f61765e03"],["E:/GitHubBlog/public/2020/09/25/20200919-0925总结/index.html","1172bbb2b202258be521c6249a861b4e"],["E:/GitHubBlog/public/2020/10/06/《Deep Keyphrase Generation》详细解析！/index.html","4151272e74c675310924a860f42fb89f"],["E:/GitHubBlog/public/2020/10/16/20201008-1017总结/index.html","8b7d3ef5f1a9e7ad56c60ce4bf343642"],["E:/GitHubBlog/public/2020/10/20/特征选择的几种方法(一)————Filter过滤法/index.html","ff6ce52fa414755d254477c204479cf5"],["E:/GitHubBlog/public/2020/10/21/PCA/index.html","621524c31b774003f416049608f4bdca"],["E:/GitHubBlog/public/2020/10/22/20201018-1023总结/index.html","37c79459c09f9d9ce01e171bebdaac53"],["E:/GitHubBlog/public/2020/10/24/20201024-1030/index.html","51cb6d6ccfd088b0abf672927997bafe"],["E:/GitHubBlog/public/2020/10/30/1030讨论记录/index.html","08fe4202e3887fa6e30b76386c2dcf97"],["E:/GitHubBlog/public/2020/11/03/20201103-总结/index.html","371c77985e5100dd05b44d8d9ffaf3a8"],["E:/GitHubBlog/public/2020/11/05/线性回归与逻辑回归/index.html","9fe85993971ed9527cbc541037714f38"],["E:/GitHubBlog/public/2020/11/07/决策树/index.html","e38b118588f2bfd27b35124e2da5ba79"],["E:/GitHubBlog/public/2020/11/13/20201107-1113总结/index.html","bef9f82c3f095bffcee12f35442c9ab4"],["E:/GitHubBlog/public/2020/11/15/20201114-119总结/index.html","7f09c6e473c439376d4014e8db26f9b0"],["E:/GitHubBlog/public/2020/11/16/手推一个SVM/index.html","0390b04b12014cd81380513574f7d1e7"],["E:/GitHubBlog/public/2020/11/27/20201127左右的总结/index.html","abbad82d755f0c6da336e3a93d1a2be0"],["E:/GitHubBlog/public/2020/12/23/第一个深度学习模型（pytorch）/index.html","e26eb595f9aa11264c55a503912c210b"],["E:/GitHubBlog/public/2021/01/02/鲍鱼最终版/index.html","dbf5b6372b7c72b1c0c79f2b87c575a5"],["E:/GitHubBlog/public/2021/01/08/GSA_NumericalExperiment/index.html","879540c744076e62be084b91796b0362"],["E:/GitHubBlog/public/2021/01/16/20210115-01总结/index.html","976da3c3da60d70bb4605cb4afbe7414"],["E:/GitHubBlog/public/2021/01/17/20200117-0118总结——OpenNMT阅读（二）/index.html","0733ce24d9d601df1de4669115e2620b"],["E:/GitHubBlog/public/2021/01/17/OpenNMT源码解析/index.html","ff19ab091fdb832355856b2450c45d2e"],["E:/GitHubBlog/public/2021/01/19/重学数据结构——线性表/index.html","ee3f4063bd771ddbff33ada9cd8fbff9"],["E:/GitHubBlog/public/2021/01/22/《深度学习入门-基于python的理论与实现》——第二章 感知机/index.html","c59a4b86823f423119bc5a6fcf8e42d4"],["E:/GitHubBlog/public/2021/01/26/title复现中的问题/index.html","f9c808268df63d6870a4e8ea5744116b"],["E:/GitHubBlog/public/2021/01/26/简单链表题/index.html","15761e2c76d7e50b107b4828b5edd851"],["E:/GitHubBlog/public/2021/01/28/DeepKeyphraseGeneration/index.html","adf7c75e56e2acbf5f6296de48d2e6e7"],["E:/GitHubBlog/public/2021/02/01/title模型详解/index.html","ce09b7223ec32ef29cbf0431771d681c"],["E:/GitHubBlog/public/2021/02/14/中等链表题/index.html","14805ea7214ba9b792f8050278db69ff"],["E:/GitHubBlog/public/2021/02/21/Teacher forcing in RNN/index.html","61265af4d2eab68a596b2ec7314ec57a"],["E:/GitHubBlog/public/2021/03/11/CopyNet代码消化记录/index.html","ea0549df4ff909b230be1fcf5c3c9ad0"],["E:/GitHubBlog/public/2021/03/11/王道机试指南习题/index.html","9123438ff82424e501c01c5e27598931"],["E:/GitHubBlog/public/2021/03/13/野生kpg代码挣扎记录/index.html","39cbb279606060987e7a158b9f5790c9"],["E:/GitHubBlog/public/2021/03/19/AcWing代码记录/index.html","99c4d4e6d9559fe572d65195e63d6bda"],["E:/GitHubBlog/public/2021/03/20/2021-03-26-week3组会/index.html","60770b9463668ed8a59dd6fc225c2b32"],["E:/GitHubBlog/public/2021/03/22/NEURAL MACHINE TRANSLATION模型/index.html","c8906dfea9de3f3697e52e75a1b6aced"],["E:/GitHubBlog/public/2021/03/23/西瓜书学习——第一章 绪论/index.html","47331da040a38b209c1d16fb196b7c5f"],["E:/GitHubBlog/public/2021/03/24/西瓜书南瓜书 第二章/index.html","9a64847a3921864b2e24843beda0dc0c"],["E:/GitHubBlog/public/2021/04/01/野生kpg代码挣扎记录2/index.html","437913cd43f71d9668c1b1fa72c9cd43"],["E:/GitHubBlog/public/2021/04/04/python爬虫爬取新浪财经个股新闻研报/index.html","00eae4b230b2b983a0a0e4c76a51d00d"],["E:/GitHubBlog/public/2021/04/04/西瓜书南瓜书 第三章/index.html","20e5a3a6ae688be3af67d63f8aa3acba"],["E:/GitHubBlog/public/2021/04/09/ip被封了怎么办/index.html","38802328129a90a7d517213387a9d6c5"],["E:/GitHubBlog/public/2021/04/12/修改代码流水账——数据预处理部分/index.html","c09c4a35aa42a19862d11e86c0d7032f"],["E:/GitHubBlog/public/2021/04/12/数据分析（机器学习）作业/index.html","114b6afd04eaa283939e9c304a62c755"],["E:/GitHubBlog/public/2021/04/15/量化文本的方法/index.html","5aea291750bdd3cb4a823bba3f714ba2"],["E:/GitHubBlog/public/2021/04/17/西瓜书南瓜书 第六章/index.html","6458466c1508adf4d14fbc846111b462"],["E:/GitHubBlog/public/archives/2020/01/index.html","14b03df1eecd502584c83c195b886599"],["E:/GitHubBlog/public/archives/2020/02/index.html","4b1a73f0c17fbad028b75906f070b4ab"],["E:/GitHubBlog/public/archives/2020/03/index.html","1031e5ab13483c80549a3aa0376c7cdd"],["E:/GitHubBlog/public/archives/2020/04/index.html","a97bc4cc222a0ec362ca20ff577216bf"],["E:/GitHubBlog/public/archives/2020/05/index.html","eaab88bc0c1a60688abc94d700ed3048"],["E:/GitHubBlog/public/archives/2020/07/index.html","200c35195488e012666ca4c4fcbf05d6"],["E:/GitHubBlog/public/archives/2020/08/index.html","7ddc00e3b2894e81042f52998a339234"],["E:/GitHubBlog/public/archives/2020/09/index.html","9b3dc7f0d340503822acf6de00373205"],["E:/GitHubBlog/public/archives/2020/10/index.html","ec07594a5d3e790cea36fc6266789e8b"],["E:/GitHubBlog/public/archives/2020/11/index.html","9235728f45390f54d7f31b8aee620c00"],["E:/GitHubBlog/public/archives/2020/12/index.html","c20b1b3e88f3181c2c1710f0cc2ea836"],["E:/GitHubBlog/public/archives/2020/index.html","6fca9a34548c343c0c01b56dfe29cc04"],["E:/GitHubBlog/public/archives/2020/page/2/index.html","ee63c39b676149c2e9dbe49ecd31b4c7"],["E:/GitHubBlog/public/archives/2020/page/3/index.html","ffde5d81791893e265deae06998ec3d8"],["E:/GitHubBlog/public/archives/2020/page/4/index.html","ff9a76c64ddaa50d41f2781fdfdbd70b"],["E:/GitHubBlog/public/archives/2020/page/5/index.html","df10785257c179f0242743735c7d8f74"],["E:/GitHubBlog/public/archives/2021/01/index.html","3905c4fc940a56cee9dd934e6e45559e"],["E:/GitHubBlog/public/archives/2021/02/index.html","47ea0621136e9ac0d4ec273ac2866f6d"],["E:/GitHubBlog/public/archives/2021/03/index.html","b50e0e1a56998bb1f97041044f40e36e"],["E:/GitHubBlog/public/archives/2021/04/index.html","e0ab96b5d17abf57740f2d9f1d3c3417"],["E:/GitHubBlog/public/archives/2021/index.html","1ed6d63284ff120ec65457b72b537131"],["E:/GitHubBlog/public/archives/2021/page/2/index.html","cf6d81e099d3657b73dbc2c742ff4f31"],["E:/GitHubBlog/public/archives/2021/page/3/index.html","788a4c4f314ff445d9f4498ccee037b0"],["E:/GitHubBlog/public/archives/index.html","5c080b789ac58caff83592eaad93629e"],["E:/GitHubBlog/public/archives/page/2/index.html","5eed28d5f44bd0d95ac98e5c02a56d5c"],["E:/GitHubBlog/public/archives/page/3/index.html","b96e4e2d1644d5f124486f1ae241e055"],["E:/GitHubBlog/public/archives/page/4/index.html","29d1c52dd441b665b3e5cf38c6c18332"],["E:/GitHubBlog/public/archives/page/5/index.html","b64beee2aee358e3547f3ef984dd2261"],["E:/GitHubBlog/public/archives/page/6/index.html","3e1c309d64c3d99f0364d2491676ae41"],["E:/GitHubBlog/public/archives/page/7/index.html","2b730112414566d341eee4aa269f9bee"],["E:/GitHubBlog/public/archives/page/8/index.html","236a19d764dc71e3afbe54cbf8e20b92"],["E:/GitHubBlog/public/assets/css/APlayer.min.css","fbe994054426fadb2dff69d824c5c67a"],["E:/GitHubBlog/public/assets/js/APlayer.min.js","8f1017e7a73737e631ff95fa51e4e7d7"],["E:/GitHubBlog/public/assets/js/Meting.min.js","bfac0368480fd344282ec018d28f173d"],["E:/GitHubBlog/public/categories/书籍学习/index.html","a954a0228a3bbe11112c8497aad1dd31"],["E:/GitHubBlog/public/css/font.css","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/css/index.css","c86084ab63c8ac8d43ecc89e8c98ae41"],["E:/GitHubBlog/public/css/post.css","d746e759ed361785cd87827baa1a6ef1"],["E:/GitHubBlog/public/css/reset.css","4fd030817d0e23aa84743dabb65009a0"],["E:/GitHubBlog/public/css/tocbot.css","e8f0173e7c5216e5359587a88a570b77"],["E:/GitHubBlog/public/css/var.css","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/img/icon-left-arrow.svg","6aa3f4fd9ad7a807fb4f7ef626dfd0c3"],["E:/GitHubBlog/public/img/icon-menu-sprite.svg","8f7f2745eb92cf88761f378b0ff98ebe"],["E:/GitHubBlog/public/index.html","a531ea868b411eb4386f58f46eed23cd"],["E:/GitHubBlog/public/js/paper.js","551d4d268064033a6b0625768834ec3e"],["E:/GitHubBlog/public/js/tocbot.js","427555b1fdee580e22f144e233498068"],["E:/GitHubBlog/public/page/2/index.html","9bae8bae584432755c15442471e76141"],["E:/GitHubBlog/public/page/3/index.html","3b9facdf0889f6e3a71dfab5bfe3cb83"],["E:/GitHubBlog/public/page/4/index.html","f39933ed5f1acce8075dfa7df46018f3"],["E:/GitHubBlog/public/page/5/index.html","479f00b03326685942724596e7b4d9a7"],["E:/GitHubBlog/public/page/6/index.html","87720f167309c2600b4b5264d708527e"],["E:/GitHubBlog/public/page/7/index.html","5f373664b1f0b896350fd84ba700123e"],["E:/GitHubBlog/public/page/8/index.html","6527927c773c6888006b2c7d12093d8d"],["E:/GitHubBlog/public/tags/Android/index.html","af23132802217ffe1c647991520692c8"],["E:/GitHubBlog/public/tags/NLP/index.html","ffbff3a207a3f0b4cc080f75dd03ca60"],["E:/GitHubBlog/public/tags/NLP/page/2/index.html","2845cdf0e1eed9353498a823cffebe6d"],["E:/GitHubBlog/public/tags/NLP/page/3/index.html","28f03d9237c2de6c3c938aee36384a9f"],["E:/GitHubBlog/public/tags/R/index.html","7bb3141cbddabb1001ee7f1642860242"],["E:/GitHubBlog/public/tags/index.html","6e7d979299e6c9983ffbf9c59de99412"],["E:/GitHubBlog/public/tags/java/index.html","95e9787c12e4fbf2fca0da5db3375867"],["E:/GitHubBlog/public/tags/java/page/2/index.html","4fcdbc8b09401761cced43fd7aa2456f"],["E:/GitHubBlog/public/tags/kpg/index.html","13a7bcfb6c08b33cc5b7e400a737c204"],["E:/GitHubBlog/public/tags/leetcode/index.html","83ea5da84d79c8b6121099abf65e8cc0"],["E:/GitHubBlog/public/tags/python/index.html","8b6913ec4c407946d89c3fc37cf9a9cc"],["E:/GitHubBlog/public/tags/pytorch/index.html","42b347e6a918473e77adc2fa8b9f6d50"],["E:/GitHubBlog/public/tags/《南瓜书》/index.html","69a95fae06f34300b3eb2fa2de5e621c"],["E:/GitHubBlog/public/tags/《西瓜书》/index.html","138947c8160c358a8a7271c65af99dd3"],["E:/GitHubBlog/public/tags/代码/index.html","1f903f7c345fbd2d12393bc7fd82ef05"],["E:/GitHubBlog/public/tags/优化方法/index.html","101545531412f253f25273e5f6777061"],["E:/GitHubBlog/public/tags/复制机制/index.html","ed175967f51f291463f6f4b870160861"],["E:/GitHubBlog/public/tags/总结/index.html","d17e71aad2648d6a8775bf94c5f20ca6"],["E:/GitHubBlog/public/tags/总结/page/2/index.html","1c3cb34c881ae3ca82bbc0d96fd9b7f1"],["E:/GitHubBlog/public/tags/数据分析/index.html","568e0e565bbdf777ec42cd96c0896b2d"],["E:/GitHubBlog/public/tags/数据挖掘/index.html","c0fc6f74a8f20ce6c172bc04c28a4215"],["E:/GitHubBlog/public/tags/数据科学实战/index.html","2569a7e301b4b6fa21a59fb4a27f52da"],["E:/GitHubBlog/public/tags/数据结构/index.html","8b6facb98909179a377976250eb89c8d"],["E:/GitHubBlog/public/tags/机器学习/index.html","412c60c24826f7babdd0ba7ab1f14665"],["E:/GitHubBlog/public/tags/机试准备/index.html","a485fe0750fbb0bcab0e43bbedea434b"],["E:/GitHubBlog/public/tags/深度学习/index.html","6536bd4422e39d0286284ff436d3e29d"],["E:/GitHubBlog/public/tags/爬虫/index.html","0782df241d7314a029c07de3693017b9"],["E:/GitHubBlog/public/tags/笔记/index.html","90397f3de50fe6f2f39e2bbfc7b3f410"],["E:/GitHubBlog/public/tags/算法/index.html","79ed43be3208e9ddeaf3c95fce11355c"],["E:/GitHubBlog/public/tags/论文/index.html","ea40548dbfd3170179835fd990199447"],["E:/GitHubBlog/public/tags/论文/page/2/index.html","1b99f19645b7ace7262195cb0b03b3df"],["E:/GitHubBlog/public/tags/论文/page/3/index.html","f81f6f6dca04286b0b72094f538ef3f7"],["E:/GitHubBlog/public/tags/读书笔记/index.html","fb0a82a43118b6b1c69e62e46b4ca800"],["E:/GitHubBlog/public/tags/项目-量化交易/index.html","5583c94b39253915b712724a2bdc3e7c"]];
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







