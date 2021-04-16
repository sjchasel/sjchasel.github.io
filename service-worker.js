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

var precacheConfig = [["E:/GitHubBlog/public/2020/01/28/hello-world/index.html","96eb99f441ee292dd225324def654b42"],["E:/GitHubBlog/public/2020/02/08/python爬虫学习总结/index.html","09ee83650b06526e65ed2e671f97d099"],["E:/GitHubBlog/public/2020/02/19/数据结构第一周-绪论/index.html","8cc7867e6a65e39d5e8111ce63fb2727"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-1/index.html","a9036e4986ffdff0702f9c2873b6708a"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-2-创建数据集/index.html","ec4b8312ef2ab631b62be56d49bb554f"],["E:/GitHubBlog/public/2020/03/01/《R语言实战》学习总结-3-图形初阶/index.html","6663e7ff146473b79190be9395574808"],["E:/GitHubBlog/public/2020/03/02/数据结构第二周-线性结构1/index.html","fd6f9e2481e376300036788b8868727c"],["E:/GitHubBlog/public/2020/03/03/《R语言实战》学习总结-4-基本数据管理/index.html","429b629bf99487f685ee6daa2e954aa5"],["E:/GitHubBlog/public/2020/03/14/数据结构第三周-线性结构2/index.html","406a7af48fe6e0cc7cde8d09fdf388ab"],["E:/GitHubBlog/public/2020/03/14/数据结构第四周-栈与队列/index.html","cd0b3a8d12bf7842897de3af7eba001d"],["E:/GitHubBlog/public/2020/03/17/leetcode记录-1、88、20/index.html","db1aefbac26c603bbcdbb0979abde296"],["E:/GitHubBlog/public/2020/03/20/温度转换程序/index.html","a6ee1588be55b4bd65a6fba814f1e352"],["E:/GitHubBlog/public/2020/03/22/数据结构第五周-递归/index.html","cdd8e943199e631e68a9012cad2daf8b"],["E:/GitHubBlog/public/2020/03/22/数据结构第六周-串与数组/index.html","488aa7aed25ee8649c4dfef9313c6fd5"],["E:/GitHubBlog/public/2020/04/02/leetcode记录-52/index.html","dfade81a1ba0085fbba11ee5b718a16a"],["E:/GitHubBlog/public/2020/04/17/Android第八周作业-每天第一次打开应用更新数据/index.html","2714a50cf65871ffe1c8a4fef43cdd24"],["E:/GitHubBlog/public/2020/05/21/attention模型/index.html","0970cb82863e041d088d55894f726bbf"],["E:/GitHubBlog/public/2020/05/21/编码器与解码器框架/index.html","eca604253b62b9d7e77140c1c78e6fd5"],["E:/GitHubBlog/public/2020/07/11/思考，快与慢/index.html","035bdceca0ea852bdb79a37a273abef2"],["E:/GitHubBlog/public/2020/07/13/20200713-0717总结/index.html","2b6980c80e0f312e331ac705a666c5b7"],["E:/GitHubBlog/public/2020/07/13/ZeroShotCeres Zero-Shot Relation Extraction/index.html","fbb685833141ae56f2d7af20ddbd7c52"],["E:/GitHubBlog/public/2020/07/20/Keyphrase Generation A Text Summarization Struggle/index.html","fa211cfd7869ba8a5752e6774595ed26"],["E:/GitHubBlog/public/2020/07/20/TPR/index.html","3c2b0bd47a863365a8f8ed9357aa05dc"],["E:/GitHubBlog/public/2020/07/23/Keyphrase Generation for Scientific Document Retrieval/index.html","5f082914d04eb2625ec710693486e148"],["E:/GitHubBlog/public/2020/07/24/20200718-0724总结/index.html","e179ea4e4df46719981df59eae851422"],["E:/GitHubBlog/public/2020/07/31/20200725-0731总结/index.html","284ce37ab01f294664af89e81e9b7355"],["E:/GitHubBlog/public/2020/08/07/20200801-0807总结/index.html","546716b230a363dc9ca01d2b1a3beb68"],["E:/GitHubBlog/public/2020/08/13/李宏毅-机器学习21/index.html","543c7a3b4563f6b6d721dc59ab2ebb36"],["E:/GitHubBlog/public/2020/08/14/20200808-0815总结/index.html","9baad0af70f8c9fc2313c60f4d8078ac"],["E:/GitHubBlog/public/2020/09/03/20200816-0903总结/index.html","0ed7700c100e20c317b4578ed2fee9e1"],["E:/GitHubBlog/public/2020/09/16/20200914-0918总结/index.html","6a0106de323e4c844058854dccbe5f06"],["E:/GitHubBlog/public/2020/09/25/20200919-0925总结/index.html","1d8765eb2bb502dec6dfa508b245e69b"],["E:/GitHubBlog/public/2020/10/06/《Deep Keyphrase Generation》详细解析！/index.html","f85f6e771284659890ad0fe8227b9f8b"],["E:/GitHubBlog/public/2020/10/16/20201008-1017总结/index.html","37cda180a11c3dadf66911521b38b37e"],["E:/GitHubBlog/public/2020/10/20/特征选择的几种方法(一)————Filter过滤法/index.html","1430b1ee0975805d88c207cd992f5715"],["E:/GitHubBlog/public/2020/10/21/PCA/index.html","6f6506f56b297bdb890038b1b5120d3a"],["E:/GitHubBlog/public/2020/10/22/20201018-1023总结/index.html","4839049905d34eb43ba278f951125571"],["E:/GitHubBlog/public/2020/10/24/20201024-1030/index.html","276f21a89b18ec77b8c7901717b4d248"],["E:/GitHubBlog/public/2020/10/30/1030讨论记录/index.html","3a89d96b06b934042646b7913b1083e3"],["E:/GitHubBlog/public/2020/11/03/20201103-总结/index.html","25ef842f911ea63982fc7e2d75e3ef72"],["E:/GitHubBlog/public/2020/11/05/线性回归与逻辑回归/index.html","ce8a4df70db5e9333cfe9e1018dabd7b"],["E:/GitHubBlog/public/2020/11/07/决策树/index.html","3b3bf7056d3be13f23db5f0a346a90df"],["E:/GitHubBlog/public/2020/11/13/20201107-1113总结/index.html","3fe4c26f6548a445cd3c3480bb1eb756"],["E:/GitHubBlog/public/2020/11/15/20201114-119总结/index.html","d18510d66f31639d2611624953d565d2"],["E:/GitHubBlog/public/2020/11/16/手推一个SVM/index.html","6110c5e2fa90cdb5a16e1a77a0568b9c"],["E:/GitHubBlog/public/2020/11/27/20201127左右的总结/index.html","9bd06d6c764879b293735d296816031c"],["E:/GitHubBlog/public/2020/12/23/第一个深度学习模型（pytorch）/index.html","9ed053a082a5cd0e5fb9b74a215455ed"],["E:/GitHubBlog/public/2021/01/02/鲍鱼最终版/index.html","a88ab004e443dd0f5e21058df164c7dc"],["E:/GitHubBlog/public/2021/01/08/GSA_NumericalExperiment/index.html","73103a4e451281832230a0d5e915501d"],["E:/GitHubBlog/public/2021/01/16/20210115-01总结/index.html","bfc1c757c8ce8c10bff5a43027234e63"],["E:/GitHubBlog/public/2021/01/17/20200117-0118总结——OpenNMT阅读（二）/index.html","c3e1281c754d100ea71159677a7719ec"],["E:/GitHubBlog/public/2021/01/17/OpenNMT源码解析/index.html","e08255f6f39386d94d0658fbad694c40"],["E:/GitHubBlog/public/2021/01/19/重学数据结构——线性表/index.html","3eece63d5b36570d65f14fb3fd375cc4"],["E:/GitHubBlog/public/2021/01/22/《深度学习入门-基于python的理论与实现》——第二章 感知机/index.html","3b7d66dfc854e0e10e240ca4cded7894"],["E:/GitHubBlog/public/2021/01/26/title复现中的问题/index.html","f825a01b968c5d55a7d0dc8519f36b01"],["E:/GitHubBlog/public/2021/01/26/简单链表题/index.html","2e7951557b0d3f02a8c43a5e77fe65fb"],["E:/GitHubBlog/public/2021/01/28/DeepKeyphraseGeneration/index.html","631f20ba85061a43c02d1a144cf7f50c"],["E:/GitHubBlog/public/2021/02/01/title模型详解/index.html","beb39e7580a7a5869cb0251ac7ea3d3c"],["E:/GitHubBlog/public/2021/02/14/中等链表题/index.html","3a240f5960358fa34a78fcfc1ed00cb3"],["E:/GitHubBlog/public/2021/02/21/Teacher forcing in RNN/index.html","e1a1910ed60feadb8f6b2d676ef6d631"],["E:/GitHubBlog/public/2021/03/11/CopyNet代码消化记录/index.html","306baacbbb2a8cf8022e085ffb1a3548"],["E:/GitHubBlog/public/2021/03/11/王道机试指南习题/index.html","2f2b55fc206551f8d28b9daf6e19c055"],["E:/GitHubBlog/public/2021/03/13/野生kpg代码挣扎记录/index.html","58e4f75183855add1d0b8f176bb083d3"],["E:/GitHubBlog/public/2021/03/19/AcWing代码记录/index.html","b7e6f6f0af7e69f5276eae2f17e821fe"],["E:/GitHubBlog/public/2021/03/20/2021-03-26-week3组会/index.html","24bc43887c6b5530bf89c99505ce4e20"],["E:/GitHubBlog/public/2021/03/22/NEURAL MACHINE TRANSLATION模型/index.html","dd17647a6844e902b8ef32f533fcb070"],["E:/GitHubBlog/public/2021/03/23/西瓜书学习——第一章 绪论/index.html","8c96c6f1d1fb51186579c38b9c4c9a1d"],["E:/GitHubBlog/public/2021/03/24/西瓜书南瓜书 第二章/index.html","b6578b39057b7cce04ed4037120eeb93"],["E:/GitHubBlog/public/2021/04/01/野生kpg代码挣扎记录2/index.html","89126298d99197701a4c146e52a8a47a"],["E:/GitHubBlog/public/2021/04/04/python爬虫爬取新浪财经个股新闻研报/index.html","4438e956a16c8e1a748815883c6b2a70"],["E:/GitHubBlog/public/2021/04/04/西瓜书南瓜书 第三章/index.html","847cef0cbf399d935d12ea67f4014f24"],["E:/GitHubBlog/public/2021/04/09/ip被封了怎么办/index.html","c6ab89ab6fe34b879999dde01c11a0a9"],["E:/GitHubBlog/public/2021/04/11/20210409-0416一周总结/index.html","abbf5bf5fbe8d112c42253bfce0f5478"],["E:/GitHubBlog/public/2021/04/12/改代码的艰难过程/index.html","72618546279cc05537ff9ab787ec429e"],["E:/GitHubBlog/public/2021/04/12/数据分析（机器学习）作业/index.html","30fc625cda7c9b829a388f11e4993202"],["E:/GitHubBlog/public/2021/04/15/量化文本的方法/index.html","22210f5ed4f25e8979ec2febf6eb52c0"],["E:/GitHubBlog/public/archives/2020/01/index.html","e6b7b918ae92cb6c67d5226077524c2f"],["E:/GitHubBlog/public/archives/2020/02/index.html","56d79debec1c5432c87e43c2aa0fc083"],["E:/GitHubBlog/public/archives/2020/03/index.html","87c30173c0cf70870f303c34536bfb68"],["E:/GitHubBlog/public/archives/2020/04/index.html","5d68528d69841fe4b2ed7432377a238d"],["E:/GitHubBlog/public/archives/2020/05/index.html","dc3c2cefb2d512898eb73b987fd40fe0"],["E:/GitHubBlog/public/archives/2020/07/index.html","14254be00d0dae23e68d93e8c20ac395"],["E:/GitHubBlog/public/archives/2020/08/index.html","97d80a3dd1cdabe0cb6fab21853ffdbd"],["E:/GitHubBlog/public/archives/2020/09/index.html","b6c05e08c4f3d511e188fead0da3b493"],["E:/GitHubBlog/public/archives/2020/10/index.html","adb3f631e16c8bae247f642ec0ca833c"],["E:/GitHubBlog/public/archives/2020/11/index.html","ac86ba4b1b9d65d983ae165f0a104f36"],["E:/GitHubBlog/public/archives/2020/12/index.html","241fa8c41b1d17082802af53cb60d1d6"],["E:/GitHubBlog/public/archives/2020/index.html","f55749b3b8f01b81de0e20893cdf7305"],["E:/GitHubBlog/public/archives/2020/page/2/index.html","fe3da1aeaa9cf020d3b7947b06caa545"],["E:/GitHubBlog/public/archives/2020/page/3/index.html","ed52afe3e2dc1367e916cfecc586c82c"],["E:/GitHubBlog/public/archives/2020/page/4/index.html","6df7b7b9230cb7c6e04ac6d7ee3e7622"],["E:/GitHubBlog/public/archives/2020/page/5/index.html","ac23e8b1bd7f1c58737def5c7e66f28f"],["E:/GitHubBlog/public/archives/2021/01/index.html","23c3dcfe6c6c6261d93c0900f8c66b5c"],["E:/GitHubBlog/public/archives/2021/02/index.html","f86e167d6918ba4c1304d69778cfe403"],["E:/GitHubBlog/public/archives/2021/03/index.html","f1dfbb4b71634af42f00685a02c5d425"],["E:/GitHubBlog/public/archives/2021/04/index.html","f6a350381485dd9b7e5d1752cf2005ce"],["E:/GitHubBlog/public/archives/2021/index.html","80dc48e2cc479f71b380e5bd07bd8fa3"],["E:/GitHubBlog/public/archives/2021/page/2/index.html","2c5f69c6569604ea497a52f5f48a3d70"],["E:/GitHubBlog/public/archives/2021/page/3/index.html","8cb35841c21f0dec5cc045163f915506"],["E:/GitHubBlog/public/archives/index.html","b454d002a3e063a2ef870e03f17369e1"],["E:/GitHubBlog/public/archives/page/2/index.html","d4179ef8a742571c41d9e95610c5f4dc"],["E:/GitHubBlog/public/archives/page/3/index.html","af5ed9e36ea11387963e9f26a12ce389"],["E:/GitHubBlog/public/archives/page/4/index.html","f4392a6a047c7261b33329af1a4bfb10"],["E:/GitHubBlog/public/archives/page/5/index.html","02cf60d4ac0be61d26857022080b849c"],["E:/GitHubBlog/public/archives/page/6/index.html","06391855f0f962d166db7f70ddad167c"],["E:/GitHubBlog/public/archives/page/7/index.html","aa0c4c433625cb2bb9f72fc84b71fea4"],["E:/GitHubBlog/public/archives/page/8/index.html","d02cd46c1f7a52d9fc178eaa76e6106a"],["E:/GitHubBlog/public/assets/css/APlayer.min.css","fbe994054426fadb2dff69d824c5c67a"],["E:/GitHubBlog/public/assets/js/APlayer.min.js","8f1017e7a73737e631ff95fa51e4e7d7"],["E:/GitHubBlog/public/assets/js/Meting.min.js","bfac0368480fd344282ec018d28f173d"],["E:/GitHubBlog/public/css/font.css","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/css/index.css","c86084ab63c8ac8d43ecc89e8c98ae41"],["E:/GitHubBlog/public/css/post.css","d746e759ed361785cd87827baa1a6ef1"],["E:/GitHubBlog/public/css/reset.css","4fd030817d0e23aa84743dabb65009a0"],["E:/GitHubBlog/public/css/tocbot.css","e8f0173e7c5216e5359587a88a570b77"],["E:/GitHubBlog/public/css/var.css","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/img/icon-left-arrow.svg","6aa3f4fd9ad7a807fb4f7ef626dfd0c3"],["E:/GitHubBlog/public/img/icon-menu-sprite.svg","8f7f2745eb92cf88761f378b0ff98ebe"],["E:/GitHubBlog/public/index.html","56fdebf1a7db959c1bcfa8908eb3d5eb"],["E:/GitHubBlog/public/js/paper.js","551d4d268064033a6b0625768834ec3e"],["E:/GitHubBlog/public/js/tocbot.js","427555b1fdee580e22f144e233498068"],["E:/GitHubBlog/public/page/2/index.html","d7784569ab8c5b89b94bca1edc359eff"],["E:/GitHubBlog/public/page/3/index.html","23306a59c8f68276910a3047df1045b0"],["E:/GitHubBlog/public/page/4/index.html","90e4334a5362cd99ff8375e28d8b152b"],["E:/GitHubBlog/public/page/5/index.html","36672ccb3350adf0bcff4297255ce7bf"],["E:/GitHubBlog/public/page/6/index.html","2af3feaa74dd04b81ddb72fccba44b7f"],["E:/GitHubBlog/public/page/7/index.html","3bf933dc340fb41813319a6548a10a7f"],["E:/GitHubBlog/public/page/8/index.html","990aeadb5f939ebc158d4088daddae16"],["E:/GitHubBlog/public/tags/Android/index.html","7a7d08b6afc8e90693db1623222efdcb"],["E:/GitHubBlog/public/tags/NLP/index.html","c9bd36027f5504851a728038f305b92b"],["E:/GitHubBlog/public/tags/NLP/page/2/index.html","0bbc36c0bba237ce48203f27b927906c"],["E:/GitHubBlog/public/tags/NLP/page/3/index.html","3518d95dfe78e90b8286cf3803f5549b"],["E:/GitHubBlog/public/tags/R/index.html","4fbedb73f69c463452636d35b7970e1d"],["E:/GitHubBlog/public/tags/index.html","0fdd5a678e39b66cb118552fe9b64e60"],["E:/GitHubBlog/public/tags/java/index.html","cf4cbdb7ac7cb97e1580baf077ac832a"],["E:/GitHubBlog/public/tags/java/page/2/index.html","1caa50eef1903b224cc93dea272fb1d8"],["E:/GitHubBlog/public/tags/leetcode/index.html","3f74c1ca9919e9d2d29ba7c7bb0f3f64"],["E:/GitHubBlog/public/tags/python/index.html","885c303785180decdf200d579dcf1fdd"],["E:/GitHubBlog/public/tags/pytorch/index.html","5f5bad57113cc345e0052a205e021222"],["E:/GitHubBlog/public/tags/代码/index.html","849922ba4bd7d5e0880940a491a938e4"],["E:/GitHubBlog/public/tags/优化方法/index.html","15bace2324f9ffac9273254aac91d8af"],["E:/GitHubBlog/public/tags/总结/index.html","ba54122e25980b6a6bda27c72673fc31"],["E:/GitHubBlog/public/tags/总结/page/2/index.html","daec15f81d30df0e0e97d8f46c0cd69a"],["E:/GitHubBlog/public/tags/数据分析/index.html","8d71cf9631fdbe5f1f72765c81cc51a3"],["E:/GitHubBlog/public/tags/数据挖掘/index.html","efb4ebdea5b193942bbd0a0e1bf5630e"],["E:/GitHubBlog/public/tags/数据科学实战/index.html","32599c0a85c815b413aac8fc58407d23"],["E:/GitHubBlog/public/tags/数据结构/index.html","2b8d3227969dd3a17068f0dafc74da7a"],["E:/GitHubBlog/public/tags/机器学习/index.html","16b528dcdd68232678ca9dcbb3a69f6a"],["E:/GitHubBlog/public/tags/深度学习/index.html","02f0a37f3b6d3671e1e787a5daf7cee1"],["E:/GitHubBlog/public/tags/爬虫/index.html","f1df860b0b71db4cbf86a663ea2c71c1"],["E:/GitHubBlog/public/tags/笔记/index.html","6f53a9c24c16089a50c0ed55860b8a8d"],["E:/GitHubBlog/public/tags/论文/index.html","346e63787541442d27f3abd2349de202"],["E:/GitHubBlog/public/tags/论文/page/2/index.html","45efc2bb46a4ac02ddc6f2f40ffee327"],["E:/GitHubBlog/public/tags/论文/page/3/index.html","29cb1da2435e947e5e333f5510260401"],["E:/GitHubBlog/public/tags/读书笔记/index.html","fbec37342b6d4ae4d43241e77fa321bf"],["E:/GitHubBlog/public/tags/量化交易/index.html","c00cabbb672ab2dcd0a394853ede199f"]];
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







