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

var precacheConfig = [["E:/GitHubBlog/public/2020/01/28/hello-world/index.html","678f900f639cb0e4562162700da44f89"],["E:/GitHubBlog/public/2020/02/08/python爬虫学习总结/index.html","e1a5e99f27f422026b9d92c864759923"],["E:/GitHubBlog/public/2020/02/19/数据结构第一周-绪论/index.html","9ec24cf395c34ec52b9d0048d02993c3"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-1/index.html","f09236cb8ecc503f8c749142f9927949"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-2-创建数据集/index.html","c9dea03fd5be74d71f4f0135984410e1"],["E:/GitHubBlog/public/2020/03/01/《R语言实战》学习总结-3-图形初阶/index.html","c958c96b560c4f162c220479a9790672"],["E:/GitHubBlog/public/2020/03/02/数据结构第二周-线性结构1/index.html","955b26dd8bf117f5f40a90671bf81a5b"],["E:/GitHubBlog/public/2020/03/03/《R语言实战》学习总结-4-基本数据管理/index.html","5d77b18a4d4154524d095631b9ba1c80"],["E:/GitHubBlog/public/2020/03/14/数据结构第三周-线性结构2/index.html","1bd56184c7454b5c278c8bba8337d2c0"],["E:/GitHubBlog/public/2020/03/14/数据结构第四周-栈与队列/index.html","6fe3cca6125d67b3c6e7b51000c23051"],["E:/GitHubBlog/public/2020/03/17/leetcode记录-1、88、20/index.html","b481e8ecab06ee50598f3f684ac7a75d"],["E:/GitHubBlog/public/2020/03/20/温度转换程序/index.html","c8bdedaf2c14c32f6f3f285a31d3d929"],["E:/GitHubBlog/public/2020/03/22/数据结构第五周-递归/index.html","e69e9daa27b0535e2fcd3ad2931a5eb8"],["E:/GitHubBlog/public/2020/03/22/数据结构第六周-串与数组/index.html","0614c3ce96f8025cd2c654c488829552"],["E:/GitHubBlog/public/2020/04/02/leetcode记录-52/index.html","f68b7a99a15bb54f0f371b8f56d99e3d"],["E:/GitHubBlog/public/2020/04/17/Android第八周作业-每天第一次打开应用更新数据/index.html","70da0e5eeacef5f71f42eab412d9988a"],["E:/GitHubBlog/public/2020/05/21/attention模型/index.html","2a34674197d4ccc9f3f6a01b2395a0b8"],["E:/GitHubBlog/public/2020/05/21/编码器与解码器框架/index.html","9b33a38450075e812672d4da58c45d34"],["E:/GitHubBlog/public/2020/07/11/思考，快与慢/index.html","46e8d150e7d8c01494497716f6e76456"],["E:/GitHubBlog/public/2020/07/13/20200713-0717总结/index.html","1c0e0912f76d1b8ea6197267b0a7009c"],["E:/GitHubBlog/public/2020/07/13/ZeroShotCeres Zero-Shot Relation Extraction/index.html","6d99ede0534915e0b8a6a0cb84733336"],["E:/GitHubBlog/public/2020/07/20/Keyphrase Generation A Text Summarization Struggle/index.html","342dc6898987f4efd57efb729714d17e"],["E:/GitHubBlog/public/2020/07/20/TPR/index.html","64f64828b400548b819e80fd15a8da52"],["E:/GitHubBlog/public/2020/07/23/Keyphrase Generation for Scientific Document Retrieval/index.html","5aa78f7bdc29a00f72e42f6892dffe75"],["E:/GitHubBlog/public/2020/07/24/20200718-0724总结/index.html","b817be3e7a65619006d49eb3b4073818"],["E:/GitHubBlog/public/2020/07/31/20200725-0731总结/index.html","bf7c30a8d898b59f0cb2f473fd4006e0"],["E:/GitHubBlog/public/2020/08/07/20200801-0807总结/index.html","696dcaef4291a2b1a5f9ab99bb461d8b"],["E:/GitHubBlog/public/2020/08/12/逻辑回归算法/index.html","96c6ed41746a9585e0f84a3d8a8da32b"],["E:/GitHubBlog/public/2020/08/13/李宏毅-机器学习21/index.html","c1bab910e921fbc7d8dba49b01ecffdd"],["E:/GitHubBlog/public/2020/08/14/20200808-0815总结/index.html","ef624232191bedabad754f03b0292f1b"],["E:/GitHubBlog/public/2020/09/03/20200816-0903总结/index.html","e59006f0916dd0f70910f525e47ef58b"],["E:/GitHubBlog/public/2020/09/16/20200914-0918总结/index.html","2d5929674b9ed53654d29ab18bf7427b"],["E:/GitHubBlog/public/2020/09/25/20200919-0925总结/index.html","b2d00fbdefc7fa2f46db7e9ec07386e2"],["E:/GitHubBlog/public/2020/10/06/《Deep Keyphrase Generation》详细解析！/index.html","4387d79102f44c7d18abed37375117de"],["E:/GitHubBlog/public/2020/10/16/20201008-1017总结/index.html","65a9ca2524ef1dad38ff1515e61507ef"],["E:/GitHubBlog/public/2020/10/20/特征选择的几种方法(一)————Filter过滤法/index.html","4dfab8995f2061e448181f4e02d13b03"],["E:/GitHubBlog/public/2020/10/21/PCA/index.html","2807cef7e4447f8d2e1a90c4995b33ca"],["E:/GitHubBlog/public/2020/10/22/20201018-1023总结/index.html","3304bd8c39f6e0cb279a6f6681a62a20"],["E:/GitHubBlog/public/2020/10/24/20201024-1030/index.html","61caea1fd8e1e7e3c7e1dcd4e76cf072"],["E:/GitHubBlog/public/2020/10/30/1030讨论记录/index.html","8de7e137545bdb1394744fc3914be414"],["E:/GitHubBlog/public/2020/11/03/20201103-总结/index.html","a619d2a277800bff30d120f785aa4283"],["E:/GitHubBlog/public/2020/11/05/线性回归与逻辑回归/index.html","63191ef9d16e76c06b742154e43cb071"],["E:/GitHubBlog/public/2020/11/07/决策树/index.html","04f8fecfeb89a3c75460c6f972126a3e"],["E:/GitHubBlog/public/2020/11/13/20201107-1113总结/index.html","023f2b5dda0c0d8e09dbaa32b10f1b70"],["E:/GitHubBlog/public/2020/11/15/20201114-119总结/index.html","2c27c188b72b048ee2977f74f2ca6efa"],["E:/GitHubBlog/public/2020/11/16/手推一个SVM/index.html","4db33a732fed4311eaf07934c163c03e"],["E:/GitHubBlog/public/2020/11/27/20201127左右的总结/index.html","7e4b42b57e87f3bc0baa31e5cbf390dd"],["E:/GitHubBlog/public/2020/12/23/第一个深度学习模型（pytorch）/index.html","5f85409f495a56f175875bafbf27b154"],["E:/GitHubBlog/public/2021/01/02/鲍鱼最终版/index.html","c068411308bc134a12aa104083922146"],["E:/GitHubBlog/public/2021/01/08/GSA_NumericalExperiment/index.html","34e1dd850bcd25d0ea1f01bc68c55940"],["E:/GitHubBlog/public/2021/01/16/20210115-01总结/index.html","868b144ac46c85b9920c4510d0eee35b"],["E:/GitHubBlog/public/2021/01/17/20200117-0118总结——OpenNMT阅读（二）/index.html","3a08615f6403586afaa83e907af11f14"],["E:/GitHubBlog/public/2021/01/17/OpenNMT源码解析/index.html","d02b3ead3d7a8c0e4aab4dd617445bbf"],["E:/GitHubBlog/public/2021/01/19/重学数据结构——线性表/index.html","ea975fd3aac0aefeddd6f8d7f4cb5c9e"],["E:/GitHubBlog/public/2021/01/22/《深度学习入门-基于python的理论与实现》——第二章 感知机/index.html","3203ca67f6d68f2ec62553d70371a66f"],["E:/GitHubBlog/public/2021/01/22/代码理论/index.html","14575f85acfd72eef1f5cdde3d775eec"],["E:/GitHubBlog/public/2021/01/26/title复现中的问题/index.html","7b69e3cb35296787120c832d69f9ceaa"],["E:/GitHubBlog/public/2021/01/26/简单链表题/index.html","e12bf9791a266b7475f3dfae6cdc3310"],["E:/GitHubBlog/public/2021/01/28/DeepKeyphraseGeneration/index.html","c400bb34317633420966392ff48aa39b"],["E:/GitHubBlog/public/2021/02/01/中等链表题/index.html","37875d5fde5ff149e67b351c4ebaa7aa"],["E:/GitHubBlog/public/archives/2020/01/index.html","e91e886435fcc6dd4e13aef415a043d6"],["E:/GitHubBlog/public/archives/2020/02/index.html","087afe8525dc1c88a3967bbba638b107"],["E:/GitHubBlog/public/archives/2020/03/index.html","f430ed4685e23061e41898a3904d8303"],["E:/GitHubBlog/public/archives/2020/04/index.html","ec97f2c764ed4b77c0c85703ff4d4084"],["E:/GitHubBlog/public/archives/2020/05/index.html","d47584fc5382b1aceacf039bb90b23d8"],["E:/GitHubBlog/public/archives/2020/07/index.html","cdb2a9669f9186426e70cf353af0084f"],["E:/GitHubBlog/public/archives/2020/08/index.html","751ac09409f2a989f3e72fed91096645"],["E:/GitHubBlog/public/archives/2020/09/index.html","f9f42a871565d988ac881a1096372779"],["E:/GitHubBlog/public/archives/2020/10/index.html","737702425bc92604a4fd5d6026700763"],["E:/GitHubBlog/public/archives/2020/11/index.html","a361074af5c46a86775d8ff8d66c907a"],["E:/GitHubBlog/public/archives/2020/12/index.html","23e0b792e66831ca841004c3410d3f4e"],["E:/GitHubBlog/public/archives/2020/index.html","a0ab1069b37766e0fd1f22ca8a9a022d"],["E:/GitHubBlog/public/archives/2020/page/2/index.html","8a8acb5f3f0feb5a90503f25266da92d"],["E:/GitHubBlog/public/archives/2020/page/3/index.html","ae065f490686a6a999df90f4a98ee63a"],["E:/GitHubBlog/public/archives/2020/page/4/index.html","354a8e97878bec5e9f182cf91c709cf5"],["E:/GitHubBlog/public/archives/2020/page/5/index.html","1780a6be017dde39b8f5ae1ddebea096"],["E:/GitHubBlog/public/archives/2021/01/index.html","3137b62b71914eedd313d73fb73ea235"],["E:/GitHubBlog/public/archives/2021/01/page/2/index.html","7fc1183f451f46a7cfc17f0e61d9db66"],["E:/GitHubBlog/public/archives/2021/02/index.html","a5e2284d178d5b84c86667a7a9353e3d"],["E:/GitHubBlog/public/archives/2021/index.html","96cd8c9219253cec615c99fd83bfbd2f"],["E:/GitHubBlog/public/archives/2021/page/2/index.html","c39f5928adc21982546c56484d41e88e"],["E:/GitHubBlog/public/archives/index.html","33f0988183e3dbe84de07f105201d93e"],["E:/GitHubBlog/public/archives/page/2/index.html","201559778409b60a5a3799ad246ff5bb"],["E:/GitHubBlog/public/archives/page/3/index.html","9d42c26fc884aefff0f07bfee04d6331"],["E:/GitHubBlog/public/archives/page/4/index.html","1c3825f962afd797d0d7dbfe986bba7e"],["E:/GitHubBlog/public/archives/page/5/index.html","35cbf074dacb48ba5c722b01422dbee0"],["E:/GitHubBlog/public/archives/page/6/index.html","29f02b0a93ac09b53844b765daaffd37"],["E:/GitHubBlog/public/assets/css/APlayer.min.css","fbe994054426fadb2dff69d824c5c67a"],["E:/GitHubBlog/public/assets/js/APlayer.min.js","8f1017e7a73737e631ff95fa51e4e7d7"],["E:/GitHubBlog/public/assets/js/Meting.min.js","bfac0368480fd344282ec018d28f173d"],["E:/GitHubBlog/public/css/404.css","b1bb50e1fabe41adcec483f6427fb3aa"],["E:/GitHubBlog/public/css/index.css","2389364df340948a5b822422956c2c87"],["E:/GitHubBlog/public/css/var.css","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/img/algolia.svg","fd40b88ac5370a5353a50b8175c1f367"],["E:/GitHubBlog/public/img/avatar.png","6cc4a809d23e3d8946a299ae4ce4e4cd"],["E:/GitHubBlog/public/index.html","bef9fa3080149c99d84bd03bd0afb438"],["E:/GitHubBlog/public/js/copy.js","45aae54bf2e43ac27ecc41eb5e0bacf7"],["E:/GitHubBlog/public/js/fancybox.js","f84d626654b9bbc05720952b3effe062"],["E:/GitHubBlog/public/js/fireworks.js","35933ce61c158ef9c33b5e089fe757ac"],["E:/GitHubBlog/public/js/head.js","347edd99f8e3921b45fa617e839d8182"],["E:/GitHubBlog/public/js/hexo-theme-melody.js","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/js/katex.js","d971ba8b8dee1120ef66744b89cfd2b1"],["E:/GitHubBlog/public/js/scroll.js","a12ad7e37b9e325f8da3d0523d3757c7"],["E:/GitHubBlog/public/js/search/algolia.js","53160985d32d6fd66d98aa0e05b081ac"],["E:/GitHubBlog/public/js/search/local-search.js","1565b508bd866ed6fbd724a3858af5d8"],["E:/GitHubBlog/public/js/sidebar.js","d24db780974e661198eeb2e45f20a28f"],["E:/GitHubBlog/public/js/third-party/anime.min.js","9b4bbe6deb700e1c3606eab732f5eea5"],["E:/GitHubBlog/public/js/third-party/canvas-ribbon.js","09c181544ddff1db701db02ac30453e0"],["E:/GitHubBlog/public/js/third-party/jquery.fancybox.min.js","3c9fa1c1199cd4f874d855ecb1641335"],["E:/GitHubBlog/public/js/third-party/jquery.min.js","c9f5aeeca3ad37bf2aa006139b935f0a"],["E:/GitHubBlog/public/js/third-party/reveal/head.min.js","aad121203010122e05f1766d92385214"],["E:/GitHubBlog/public/js/third-party/reveal/highlight.min.js","44594243bec43813a16371af8fe7e105"],["E:/GitHubBlog/public/js/third-party/reveal/markdown.min.js","7ec4cef5a7fe3f0bf0eb4dc6d7bca114"],["E:/GitHubBlog/public/js/third-party/reveal/marked.min.js","c2a88705e206d71dc21fdc4445349127"],["E:/GitHubBlog/public/js/third-party/reveal/math.min.js","0a278fee2e57c530ab07f7d2d9ea8d96"],["E:/GitHubBlog/public/js/third-party/reveal/notes.min.js","89a0dfae4d706f9c75b317f686c3aa14"],["E:/GitHubBlog/public/js/third-party/reveal/reveal.min.js","8988419d67efb5fe93e291a357e26ec9"],["E:/GitHubBlog/public/js/third-party/reveal/zoom.min.js","9791f96e63e7d534cba2b67d4bda0419"],["E:/GitHubBlog/public/js/third-party/velocity.min.js","64da069aba987ea0512cf610600a56d1"],["E:/GitHubBlog/public/js/third-party/velocity.ui.min.js","c8ca438424a080620f7b2f4ee4b0fff1"],["E:/GitHubBlog/public/js/transition.js","911db4268f0f6621073afcced9e1bfef"],["E:/GitHubBlog/public/js/utils.js","3ff3423d966a1c351e9867813b3f6d36"],["E:/GitHubBlog/public/page/2/index.html","d857f148b2ece9466ec6bdc3e5de0b8a"],["E:/GitHubBlog/public/page/3/index.html","7a1da66b354cb2ef251d05ebe9b63716"],["E:/GitHubBlog/public/page/4/index.html","b02e219aa7b8fe745d7808aa2b7dc88b"],["E:/GitHubBlog/public/page/5/index.html","287b3d4edc5b136637bab781f22a5eff"],["E:/GitHubBlog/public/page/6/index.html","c75a9b0afdf59143457366acb463180d"],["E:/GitHubBlog/public/tags/Android/index.html","c810b2b3a6cb3bbb8d360f119bdaddde"],["E:/GitHubBlog/public/tags/NLP/index.html","b325e6a82993457fb758b592b584906a"],["E:/GitHubBlog/public/tags/NLP/page/2/index.html","a54fec5acea9049b835e18d8b7118de5"],["E:/GitHubBlog/public/tags/NLP/page/3/index.html","756d1819cd5e087b9cbf9fa13bd73298"],["E:/GitHubBlog/public/tags/R/index.html","1fea3e7c21c3d7b1f1c0f3ef108c44d8"],["E:/GitHubBlog/public/tags/index.html","2c0f1d42567336b1d113e2fb11d33bee"],["E:/GitHubBlog/public/tags/java/index.html","29a482f34cdbf122efb3011420e58a56"],["E:/GitHubBlog/public/tags/java/page/2/index.html","50906cf7276406d574102a9ae1ed1240"],["E:/GitHubBlog/public/tags/leetcode/index.html","de777239fbfbcc475cdf42686e083280"],["E:/GitHubBlog/public/tags/python/index.html","5828ad831cde239c7e56024ecc9e10a8"],["E:/GitHubBlog/public/tags/pytorch/index.html","1ee62f4b23bf95b91a062aa67caa20fe"],["E:/GitHubBlog/public/tags/优化方法/index.html","cd7814c25eb8e9c9a4066d3890fd1fbd"],["E:/GitHubBlog/public/tags/总结/index.html","1db59a576ce4d5ec87c4c803cce3339f"],["E:/GitHubBlog/public/tags/总结/page/2/index.html","021da69c6e84f207c7fdc44d8d9e8701"],["E:/GitHubBlog/public/tags/数据分析/index.html","91f15355c4ae899d2d3d813e8b968645"],["E:/GitHubBlog/public/tags/数据挖掘/index.html","bb8906254e015f0f3156ae5d126b837e"],["E:/GitHubBlog/public/tags/数据结构/index.html","3021abe27bb7546b3b12b1f7356a6c37"],["E:/GitHubBlog/public/tags/机器学习/index.html","7a381ad588f6c02b966308b56c1a3e56"],["E:/GitHubBlog/public/tags/深度学习/index.html","ee0aab67c9c0703fc7c11d43bfc3d4f7"],["E:/GitHubBlog/public/tags/爬虫/index.html","dbf7cd43582402883d8513018d9593ae"],["E:/GitHubBlog/public/tags/笔记/index.html","df44ba8e4d9e5cbf73cb56c254a49437"],["E:/GitHubBlog/public/tags/论文/index.html","cb01a4689fbdb9baeab4714fd4b6036b"],["E:/GitHubBlog/public/tags/论文/page/2/index.html","0b8444c7ae9bca22343352fa66f1b9a6"],["E:/GitHubBlog/public/tags/论文/page/3/index.html","9677fb352768135f7c99f2fdf8412eeb"],["E:/GitHubBlog/public/tags/读书笔记/index.html","72a385d53c90cdb70707dc0439889881"]];
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







