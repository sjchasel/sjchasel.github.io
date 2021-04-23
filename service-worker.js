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

var precacheConfig = [["E:/GitHubBlog/public/2020/01/28/hello-world/index.html","cc3afe6cecc30d920ae1f2ee254a61f4"],["E:/GitHubBlog/public/2020/02/08/python爬虫学习总结/index.html","3752ba4aaf917a17b615412b6e24deaa"],["E:/GitHubBlog/public/2020/02/19/数据结构第一周-绪论/index.html","e11e7f281f96e3a7b76b0622fb45fe62"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-1/index.html","ccb92633c679d7b5f202171f9eeeaf55"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-2-创建数据集/index.html","37597ec2c8d7f15a4575edada9cccaf4"],["E:/GitHubBlog/public/2020/03/01/《R语言实战》学习总结-3-图形初阶/index.html","824f26df460a688237df6cfad5c4226a"],["E:/GitHubBlog/public/2020/03/02/数据结构第二周-线性结构1/index.html","ddd73156a26008b80d4b13b28910ff4d"],["E:/GitHubBlog/public/2020/03/03/《R语言实战》学习总结-4-基本数据管理/index.html","a9a2b8fee424ea1ebeb3906fb663195f"],["E:/GitHubBlog/public/2020/03/14/数据结构第三周-线性结构2/index.html","97204542d5f9617db0aa2168b12cc30f"],["E:/GitHubBlog/public/2020/03/14/数据结构第四周-栈与队列/index.html","af7b6a3a69417befe4c829b4591dedee"],["E:/GitHubBlog/public/2020/03/17/leetcode记录-1、88、20/index.html","60cc42837dfc0e14e6896726a7184a33"],["E:/GitHubBlog/public/2020/03/20/温度转换程序/index.html","8c51ffc0cb3cd3f1016da0fe64c02026"],["E:/GitHubBlog/public/2020/03/22/数据结构第五周-递归/index.html","7c4821aaaf3a86cb6c44851b3ecf01de"],["E:/GitHubBlog/public/2020/03/22/数据结构第六周-串与数组/index.html","7fb2e2906f6a47ed074e6db4a4e7c987"],["E:/GitHubBlog/public/2020/04/02/leetcode记录-52/index.html","9a7818b4b7e005c4e588874cdbae4d44"],["E:/GitHubBlog/public/2020/04/17/Android第八周作业-每天第一次打开应用更新数据/index.html","5a2ef3a4deb6c03cbd9285c633919404"],["E:/GitHubBlog/public/2020/05/21/attention模型/index.html","bbf113c19883cdb28abe6544317c51ea"],["E:/GitHubBlog/public/2020/05/21/编码器与解码器框架/index.html","b6fda5a05987fa057b141a97ebaed87c"],["E:/GitHubBlog/public/2020/07/11/思考，快与慢/index.html","a0df4bb5d82c9e53a41d6298a119356e"],["E:/GitHubBlog/public/2020/07/13/20200713-0717总结/index.html","4c16ef63d75acb7721aa6574943dbb82"],["E:/GitHubBlog/public/2020/07/13/ZeroShotCeres Zero-Shot Relation Extraction/index.html","9115d575842d4038deaa451049980168"],["E:/GitHubBlog/public/2020/07/20/Keyphrase Generation A Text Summarization Struggle/index.html","69b22d4c03a624b2bb1f3484873e3bcd"],["E:/GitHubBlog/public/2020/07/20/TPR/index.html","2605da5c211c66ee4995320cc893e50c"],["E:/GitHubBlog/public/2020/07/23/Keyphrase Generation for Scientific Document Retrieval/index.html","802d61347296be823eaba99726913f6c"],["E:/GitHubBlog/public/2020/07/24/20200718-0724总结/index.html","c1b244e906865f769ad9f864858043e9"],["E:/GitHubBlog/public/2020/07/31/20200725-0731总结/index.html","0985904c2596eb6cd2fcdfb4c61d4afb"],["E:/GitHubBlog/public/2020/08/07/20200801-0807总结/index.html","8e3f513e7f288e7715fcd247a1d89996"],["E:/GitHubBlog/public/2020/08/13/李宏毅-机器学习21/index.html","5c117399f25aed0d6b8a1553d2b3930a"],["E:/GitHubBlog/public/2020/08/14/20200808-0815总结/index.html","08937883be971c2e79c1e384a3d9b692"],["E:/GitHubBlog/public/2020/09/03/20200816-0903总结/index.html","3f8a34da77491e8c846dfc20baf84c7d"],["E:/GitHubBlog/public/2020/09/16/20200914-0918总结/index.html","8be51d1bfcebb310e3c609552cf2a011"],["E:/GitHubBlog/public/2020/09/25/20200919-0925总结/index.html","152921b31b7705d129987d8de677e525"],["E:/GitHubBlog/public/2020/10/06/《Deep Keyphrase Generation》详细解析！/index.html","f16d3348faca32b3a9d9700ba3ec1096"],["E:/GitHubBlog/public/2020/10/16/20201008-1017总结/index.html","59caae6047d1791ebcf9ad0eefd9af98"],["E:/GitHubBlog/public/2020/10/20/特征选择的几种方法(一)————Filter过滤法/index.html","12cb842860cccf723acc385b2517d045"],["E:/GitHubBlog/public/2020/10/21/PCA/index.html","de9ea1f68d07111099d547ef90f056cb"],["E:/GitHubBlog/public/2020/10/22/20201018-1023总结/index.html","34f73159b17e8bd5028a67f904f14927"],["E:/GitHubBlog/public/2020/10/24/20201024-1030/index.html","1c09604ecac6dfd5997648529234e4f2"],["E:/GitHubBlog/public/2020/10/30/1030讨论记录/index.html","375fe468b7ad38bb171b6b187bb8baba"],["E:/GitHubBlog/public/2020/11/03/20201103-总结/index.html","5482fc417f20a024a24a352e82db913d"],["E:/GitHubBlog/public/2020/11/05/线性回归与逻辑回归/index.html","168c929a740985f0f869d50b069060c2"],["E:/GitHubBlog/public/2020/11/07/决策树/index.html","e82b996f4143b99ad7ea2b8a18b18687"],["E:/GitHubBlog/public/2020/11/13/20201107-1113总结/index.html","c9391d932a80ea6a08c13f5932451d77"],["E:/GitHubBlog/public/2020/11/15/20201114-119总结/index.html","596c5b446bb144e446e7de2048da686c"],["E:/GitHubBlog/public/2020/11/16/手推一个SVM/index.html","549b1a9fddb6fe1d07b432d905b3f5dc"],["E:/GitHubBlog/public/2020/11/27/20201127左右的总结/index.html","68f8a8072e672002ae9f76322d2735ee"],["E:/GitHubBlog/public/2020/12/23/第一个深度学习模型（pytorch）/index.html","e8ba83b1cb9e193122d5b561c49df205"],["E:/GitHubBlog/public/2021/01/02/鲍鱼最终版/index.html","8757c9ac5f3c8b973b1c3965e59456a6"],["E:/GitHubBlog/public/2021/01/08/GSA_NumericalExperiment/index.html","90433d70157db696ced524742cf78d9a"],["E:/GitHubBlog/public/2021/01/16/20210115-01总结/index.html","a286894d9d7a43366c9776e8fd852cfc"],["E:/GitHubBlog/public/2021/01/17/20200117-0118总结——OpenNMT阅读（二）/index.html","a07ae3c33dea1e70ea9bded155d4298c"],["E:/GitHubBlog/public/2021/01/17/OpenNMT源码解析/index.html","780b4ebb96be7d92a0ec8f6a1adcc8f9"],["E:/GitHubBlog/public/2021/01/19/重学数据结构——线性表/index.html","7e406cdd7faebf30fed124edb4d8b481"],["E:/GitHubBlog/public/2021/01/22/《深度学习入门-基于python的理论与实现》——第二章 感知机/index.html","07345ea34bd1fe2dd3089248c65f2126"],["E:/GitHubBlog/public/2021/01/26/简单链表题/index.html","77fdbb6d6d6ef5cdefedda797fa07e8d"],["E:/GitHubBlog/public/2021/01/28/DeepKeyphraseGeneration/index.html","ebe76434c2fed90dabc0cd86dc4371d6"],["E:/GitHubBlog/public/2021/02/01/title模型详解/index.html","aacdecd583319556786c3dba21596cf7"],["E:/GitHubBlog/public/2021/02/14/中等链表题/index.html","fac236a1ab571389dbbcc081e41e6507"],["E:/GitHubBlog/public/2021/02/21/Teacher forcing in RNN/index.html","1e6080b6c8995db124f4ffa85b6725cc"],["E:/GitHubBlog/public/2021/03/11/CopyNet代码消化记录/index.html","ec6cf5c79b434b36aa6bb75e75fd1343"],["E:/GitHubBlog/public/2021/03/11/王道机试指南习题/index.html","8839f6735b7aa9273626290eae8990c7"],["E:/GitHubBlog/public/2021/03/19/AcWing代码记录/index.html","82b2576ab369060c44e64da03bd42b59"],["E:/GitHubBlog/public/2021/03/20/2021-03-26-week3组会/index.html","ed3281903b9b513882831dbf5e91c1c0"],["E:/GitHubBlog/public/2021/03/22/NEURAL MACHINE TRANSLATION模型/index.html","dd2962626265c1e21a6189f7d60cb212"],["E:/GitHubBlog/public/2021/03/23/西瓜书学习——第一章 绪论/index.html","331b1e674765fe855ec09f998aeb9260"],["E:/GitHubBlog/public/2021/03/24/西瓜书南瓜书 第二章/index.html","4f7bf5b0c0d1af58a91a8ecc9f628aa5"],["E:/GitHubBlog/public/2021/04/04/python爬虫爬取新浪财经个股新闻研报/index.html","f7509dd919c47a3c4c8f07a7465e2dc7"],["E:/GitHubBlog/public/2021/04/04/西瓜书南瓜书 第三章/index.html","7b7660a5edae4c272614358a775ecbfb"],["E:/GitHubBlog/public/2021/04/09/ip被封了怎么办/index.html","67ef6e4966adc7cb13ea993ccc350f3b"],["E:/GitHubBlog/public/2021/04/12/数据分析（机器学习）作业/index.html","fd0e2faebbf741b7f8432267080e14cb"],["E:/GitHubBlog/public/2021/04/15/量化文本的方法/index.html","c4ae0acd738db352b5d67dd5282c094b"],["E:/GitHubBlog/public/2021/04/17/西瓜书南瓜书 第六章/index.html","60a37db854680abb725d9e803749e6d1"],["E:/GitHubBlog/public/2021/04/18/scrapy新浪财经的重新爬取/index.html","8e4e915a6857456dafe22da0d7e0d475"],["E:/GitHubBlog/public/2021/04/19/AcWing代码记录2/index.html","5f48137424bc6289878ded907fe68c38"],["E:/GitHubBlog/public/2021/04/20/基于情感词典的中文情感分析初探/index.html","3c21277995c80ff66e45137399134731"],["E:/GitHubBlog/public/2021/04/21/一些量化文本的方法实现/index.html","a74442f766e3bcdaf5ee0c45aae0b766"],["E:/GitHubBlog/public/archives/2020/01/index.html","3051b7159482b9dfa23a826bbb10f0a6"],["E:/GitHubBlog/public/archives/2020/02/index.html","e90b44cc4fbbc1b30d6f2af6438f946d"],["E:/GitHubBlog/public/archives/2020/03/index.html","683dc4d2ff2c0c067708378a8e60b82e"],["E:/GitHubBlog/public/archives/2020/04/index.html","329c52a3f68124005be718918a8f7d6d"],["E:/GitHubBlog/public/archives/2020/05/index.html","294e60082ab1a5033adeb1beb88f35eb"],["E:/GitHubBlog/public/archives/2020/07/index.html","ad6aab7df1b712c49ceff1533d8ceed3"],["E:/GitHubBlog/public/archives/2020/08/index.html","e79c5e58398e6260dae82819ca758903"],["E:/GitHubBlog/public/archives/2020/09/index.html","c1330114054264a8696bd6c758aa7d82"],["E:/GitHubBlog/public/archives/2020/10/index.html","c57a7096c5cf486ee31b96310a0c1f02"],["E:/GitHubBlog/public/archives/2020/11/index.html","ccd2922445ed937d954d67015ee2bcb3"],["E:/GitHubBlog/public/archives/2020/12/index.html","e6a6c9193ceb97b305d0e954555732e6"],["E:/GitHubBlog/public/archives/2020/index.html","907f2f413a89ca77c69b489ecff6d002"],["E:/GitHubBlog/public/archives/2020/page/2/index.html","4e09fb9fd4fd0c1505d6e215e48d2fb4"],["E:/GitHubBlog/public/archives/2020/page/3/index.html","c19f4e4bd448709c15b0dacdb32848e7"],["E:/GitHubBlog/public/archives/2020/page/4/index.html","3821d53757befe8e0d9370313c15b8d9"],["E:/GitHubBlog/public/archives/2020/page/5/index.html","fa00e0ee395035db971fc0c24d401a66"],["E:/GitHubBlog/public/archives/2021/01/index.html","6663eaf71b0fbf6908e0e3f48ad7bae1"],["E:/GitHubBlog/public/archives/2021/02/index.html","a9b0931b290260c583a5ec3b4c06d759"],["E:/GitHubBlog/public/archives/2021/03/index.html","8511811ce7a64735bfdffabe71c79f50"],["E:/GitHubBlog/public/archives/2021/04/index.html","76e5c3ec2b6ef6e4298e7efe71bec540"],["E:/GitHubBlog/public/archives/2021/index.html","a272043494f4b5c37d6fde65e5f2057d"],["E:/GitHubBlog/public/archives/2021/page/2/index.html","278d7694ef245b3d8c44b74b199c0dc7"],["E:/GitHubBlog/public/archives/2021/page/3/index.html","719ff818e73d80c2e51111ae293be80b"],["E:/GitHubBlog/public/archives/index.html","a1050abcb04e2d68d8212a2437d39cd3"],["E:/GitHubBlog/public/archives/page/2/index.html","3411bb22e27b61808a8cb61108c40825"],["E:/GitHubBlog/public/archives/page/3/index.html","276ed3fadd5411d3109190b960fc7a99"],["E:/GitHubBlog/public/archives/page/4/index.html","36be465e7358dde1739951fa7fbaa077"],["E:/GitHubBlog/public/archives/page/5/index.html","26e4b1e8ed88ae1d4a2f45782e9d1bc5"],["E:/GitHubBlog/public/archives/page/6/index.html","f7b97883c61c19354db66e1a394119f1"],["E:/GitHubBlog/public/archives/page/7/index.html","01dc36aa4e246a0fdead2cb753ee13a0"],["E:/GitHubBlog/public/archives/page/8/index.html","e16a7bb4fa9715922192ede222db4430"],["E:/GitHubBlog/public/assets/css/APlayer.min.css","fbe994054426fadb2dff69d824c5c67a"],["E:/GitHubBlog/public/assets/js/APlayer.min.js","8f1017e7a73737e631ff95fa51e4e7d7"],["E:/GitHubBlog/public/assets/js/Meting.min.js","bfac0368480fd344282ec018d28f173d"],["E:/GitHubBlog/public/categories/书籍学习/index.html","82deba34ffee7695ad67c9b3f622b74d"],["E:/GitHubBlog/public/css/font.css","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/css/index.css","c86084ab63c8ac8d43ecc89e8c98ae41"],["E:/GitHubBlog/public/css/post.css","d746e759ed361785cd87827baa1a6ef1"],["E:/GitHubBlog/public/css/reset.css","4fd030817d0e23aa84743dabb65009a0"],["E:/GitHubBlog/public/css/tocbot.css","e8f0173e7c5216e5359587a88a570b77"],["E:/GitHubBlog/public/css/var.css","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/img/icon-left-arrow.svg","6aa3f4fd9ad7a807fb4f7ef626dfd0c3"],["E:/GitHubBlog/public/img/icon-menu-sprite.svg","8f7f2745eb92cf88761f378b0ff98ebe"],["E:/GitHubBlog/public/index.html","03a6f33d0ff069b5fea8b64dc3a0925e"],["E:/GitHubBlog/public/js/paper.js","551d4d268064033a6b0625768834ec3e"],["E:/GitHubBlog/public/js/tocbot.js","427555b1fdee580e22f144e233498068"],["E:/GitHubBlog/public/page/2/index.html","ad431583191f66cef626ecc0ef62cb98"],["E:/GitHubBlog/public/page/3/index.html","fd4fab2c3939aba9d9f5339153152dfc"],["E:/GitHubBlog/public/page/4/index.html","47c2f7df35a57181f7eae203e0b1769b"],["E:/GitHubBlog/public/page/5/index.html","82ba7d007569d78fa560c401e89dd2da"],["E:/GitHubBlog/public/page/6/index.html","d51c915065a3e5301b4d010c3cc77319"],["E:/GitHubBlog/public/page/7/index.html","28583678682e4881a7dcdc5f85c8f878"],["E:/GitHubBlog/public/page/8/index.html","0c9ae7d24603b7fd41dc6b8b1658977f"],["E:/GitHubBlog/public/tags/Android/index.html","52016f6af5c8ec9031030c6e3580fa80"],["E:/GitHubBlog/public/tags/NLP/index.html","cabd1bf1fb23a9608b0febc4b5486adf"],["E:/GitHubBlog/public/tags/NLP/page/2/index.html","6aebda8807283cd9b8d3b1a58a47405d"],["E:/GitHubBlog/public/tags/NLP/page/3/index.html","09030ee99266dc522307ee9968060da0"],["E:/GitHubBlog/public/tags/R/index.html","a0f478ee4cbc19ce6eae326b8ab67aed"],["E:/GitHubBlog/public/tags/index.html","99874528106e71116e111e36891e9eb8"],["E:/GitHubBlog/public/tags/java/index.html","8d6bcabee13448796883a82727cadb1d"],["E:/GitHubBlog/public/tags/java/page/2/index.html","a60886a587710062833f4517ef1f2387"],["E:/GitHubBlog/public/tags/kpg/index.html","558760fce9ab01951280f624ef3d379f"],["E:/GitHubBlog/public/tags/leetcode/index.html","87686491c9d714f981ec44f737c2de80"],["E:/GitHubBlog/public/tags/python/index.html","b03954921255e7d9bcf123ea84163b7b"],["E:/GitHubBlog/public/tags/pytorch/index.html","3396d6058a781e8167eb6941c9d4e870"],["E:/GitHubBlog/public/tags/《南瓜书》/index.html","888d8133162b70d32b8a83eed5e48cb5"],["E:/GitHubBlog/public/tags/《西瓜书》/index.html","0c8e8481f10ae76f8be7e4acbf27a611"],["E:/GitHubBlog/public/tags/优化方法/index.html","a965ab545cf4d161b9ad17f271993c71"],["E:/GitHubBlog/public/tags/复制机制/index.html","911afa2afa7153bc856db9b2a847e681"],["E:/GitHubBlog/public/tags/总结/index.html","540022d88bb8475d6a56c36a4ba6ad8e"],["E:/GitHubBlog/public/tags/总结/page/2/index.html","7f5355a0c4d4241d4eb9c5bf33e0ef2c"],["E:/GitHubBlog/public/tags/数据分析/index.html","833b5475c42d2f8d08e00043a45d5ba5"],["E:/GitHubBlog/public/tags/数据挖掘/index.html","c6be382b1db8efe30c8f863568e052a7"],["E:/GitHubBlog/public/tags/数据科学实战/index.html","a4e67d878dab3d7845c7cd7dfbf32491"],["E:/GitHubBlog/public/tags/数据结构/index.html","1c0f7923530d7fba32942d1201b4aa51"],["E:/GitHubBlog/public/tags/机器学习/index.html","c1af3f7b72354b908577f46678193ddb"],["E:/GitHubBlog/public/tags/机试准备/index.html","971aac513b6dca61d016ffd4cfc21aca"],["E:/GitHubBlog/public/tags/深度学习/index.html","a0d179166942dc42e78b1aa64c856940"],["E:/GitHubBlog/public/tags/爬虫/index.html","bb8ef334c8ffd59f10751d817bdf5b43"],["E:/GitHubBlog/public/tags/笔记/index.html","dbcfecc4d14db70286a20ac5b00fb783"],["E:/GitHubBlog/public/tags/算法/index.html","2e4ca8926e445227790f0d705775ee63"],["E:/GitHubBlog/public/tags/论文/index.html","cc9148da655603ce05b02a4419fa3f4c"],["E:/GitHubBlog/public/tags/论文/page/2/index.html","685973c5abeb01eac32ee1a957df8478"],["E:/GitHubBlog/public/tags/论文/page/3/index.html","8ba743c4a2f0771bb7d9451b8bf28b8c"],["E:/GitHubBlog/public/tags/读书笔记/index.html","8a94d7a31c7e808736ff68b1f6bffeab"],["E:/GitHubBlog/public/tags/量化交易/index.html","7c82c0a8c44edae62ac6c5749092d2d5"],["E:/GitHubBlog/public/tags/项目-量化交易/index.html","b7bd2e6bea22ca980b3038f99de1b481"]];
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







