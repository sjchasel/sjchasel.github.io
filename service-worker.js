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

var precacheConfig = [["E:/GitHubBlog/public/2020/01/28/hello-world/index.html","1d269b2b5a728aaabf6e9309e11e9064"],["E:/GitHubBlog/public/2020/02/08/python爬虫学习总结/index.html","bc09926054f8ba0d701925cd9e264861"],["E:/GitHubBlog/public/2020/02/19/数据结构第一周-绪论/index.html","652446c74f5e32985bb8c638eb5d6a67"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-1/index.html","ccf4fbd55c02b9795578694aada82005"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-2-创建数据集/index.html","aae45962e9a9548c0b62094e4dd2009e"],["E:/GitHubBlog/public/2020/03/01/《R语言实战》学习总结-3-图形初阶/index.html","39ec187f88d8e201eee890c2d07b14ee"],["E:/GitHubBlog/public/2020/03/02/数据结构第二周-线性结构1/index.html","c342db5a9011db20261741aaea710b7b"],["E:/GitHubBlog/public/2020/03/03/《R语言实战》学习总结-4-基本数据管理/index.html","35a2caa6a5a61634fd54a4b69b5d8ee3"],["E:/GitHubBlog/public/2020/03/14/数据结构第三周-线性结构2/index.html","a84718b76ce5705fa24eb5cd10283a59"],["E:/GitHubBlog/public/2020/03/14/数据结构第四周-栈与队列/index.html","32833695cb8ec52baeb5d5f86ee3b54e"],["E:/GitHubBlog/public/2020/03/17/leetcode记录-1、88、20/index.html","4109e51701a8d8470380bb8de906decb"],["E:/GitHubBlog/public/2020/03/20/温度转换程序/index.html","1a8e043181a88938e7b401bbeb80efda"],["E:/GitHubBlog/public/2020/03/22/数据结构第五周-递归/index.html","4f17038426622b6b1ecd60db2a4e2c63"],["E:/GitHubBlog/public/2020/03/22/数据结构第六周-串与数组/index.html","812d183ce81b53a30997d51fcd5d323d"],["E:/GitHubBlog/public/2020/04/02/leetcode记录-52/index.html","76439c0b4241abf0cda1456c092b7404"],["E:/GitHubBlog/public/2020/04/17/Android第八周作业-每天第一次打开应用更新数据/index.html","08cfd4388e8bd789874ab389b09405ec"],["E:/GitHubBlog/public/2020/05/21/attention模型/index.html","b103a3a56c4f5d52bebd64388905e5b4"],["E:/GitHubBlog/public/2020/05/21/编码器与解码器框架/index.html","228bcc53e5a6595952949b4303229fc2"],["E:/GitHubBlog/public/2020/07/11/思考，快与慢/index.html","ccd858e1366f2e8756cd9ddf2b5b6220"],["E:/GitHubBlog/public/2020/07/13/20200713-0717总结/index.html","24397e77cd11e6027d5b2e20b2080a1b"],["E:/GitHubBlog/public/2020/07/13/ZeroShotCeres Zero-Shot Relation Extraction/index.html","646b41328284fff0941dbab9b768be14"],["E:/GitHubBlog/public/2020/07/20/Keyphrase Generation A Text Summarization Struggle/index.html","f3cb9cf5d962c344f18fd2a2c9f2441a"],["E:/GitHubBlog/public/2020/07/20/TPR/index.html","309a79392ea150f15c0ef0591681dc97"],["E:/GitHubBlog/public/2020/07/23/Keyphrase Generation for Scientific Document Retrieval/index.html","ea30887dbeedada57c698da401fef912"],["E:/GitHubBlog/public/2020/07/24/20200718-0724总结/index.html","c43e5f53ca271aef16dd7ee91db83fe8"],["E:/GitHubBlog/public/2020/07/31/20200725-0731总结/index.html","8595940740c1dde3aaf454b096fbf76a"],["E:/GitHubBlog/public/2020/08/07/20200801-0807总结/index.html","0767f7b7cac7f8dff521823861eb5a1b"],["E:/GitHubBlog/public/2020/08/12/逻辑回归算法/index.html","f09d49fd021d35dad9b9215951186309"],["E:/GitHubBlog/public/2020/08/13/李宏毅-机器学习21/index.html","f47c8eae1596a1f79ddc7fd30b19ade8"],["E:/GitHubBlog/public/2020/08/14/20200808-0815总结/index.html","bd7e2a87828491050c07d2aa9bd8dc3d"],["E:/GitHubBlog/public/2020/09/03/20200816-0903总结/index.html","fed83e509efd77de860547b96c632202"],["E:/GitHubBlog/public/2020/09/16/20200914-0918总结/index.html","0865900878fc4cc3bfbcf949b38ee30c"],["E:/GitHubBlog/public/2020/09/25/20200919-0925总结/index.html","e4289f825fc5bf1691b5b221fd43d37e"],["E:/GitHubBlog/public/2020/10/06/《Deep Keyphrase Generation》详细解析！/index.html","8b6ea0641a85a775bc1d543e1b30fdb9"],["E:/GitHubBlog/public/2020/10/16/20201008-1017总结/index.html","3544d8061e1291efecffdd86ee928d36"],["E:/GitHubBlog/public/2020/10/20/特征选择的几种方法(一)————Filter过滤法/index.html","f53419179c77cc4893869c15f697ab87"],["E:/GitHubBlog/public/2020/10/21/PCA/index.html","86af5681e355cc78b81dcdf2b18f0601"],["E:/GitHubBlog/public/2020/10/22/20201018-1023总结/index.html","5a7c2f52e6130f5485412551eebda8e3"],["E:/GitHubBlog/public/2020/10/24/20201024-1030/index.html","22355a61d5aa45239ba5b2b5dac1922b"],["E:/GitHubBlog/public/2020/10/30/1030讨论记录/index.html","b8f5fbeeaef9e7e677f170fa79fe482f"],["E:/GitHubBlog/public/2020/11/03/20201103-总结/index.html","1a000b3d308151cc0a5bcc83ed6dac44"],["E:/GitHubBlog/public/2020/11/05/线性回归与逻辑回归/index.html","838fb40d021a64fd7b78e690536a6105"],["E:/GitHubBlog/public/2020/11/07/决策树/index.html","31bda86346a1e53116c08d4535e7968e"],["E:/GitHubBlog/public/2020/11/13/20201107-1113总结/index.html","24f1da19a9a277630e2ef73d0abe1610"],["E:/GitHubBlog/public/2020/11/15/20201114-119总结/index.html","2907d437890e547992080ba44e37e0f4"],["E:/GitHubBlog/public/2020/11/16/手推一个SVM/index.html","123afaffc299e349c4590f1cdf0fa4ef"],["E:/GitHubBlog/public/2020/11/27/20201127左右的总结/index.html","103178d6dd17e01627ab251e974c26c5"],["E:/GitHubBlog/public/2020/12/23/第一个深度学习模型（pytorch）/index.html","eaa35eb9841f4fe8e35ad5b64c3fb154"],["E:/GitHubBlog/public/2021/01/02/鲍鱼最终版/index.html","a017963491b338b5fa06850fff19f9ff"],["E:/GitHubBlog/public/2021/01/08/GSA_NumericalExperiment/index.html","02213173e14dbb160b6582ddc3ef771a"],["E:/GitHubBlog/public/2021/01/16/20210115-01总结/index.html","13db9345140c25a975d538bcb90cdb6c"],["E:/GitHubBlog/public/2021/01/17/20200117-0118总结——OpenNMT阅读（二）/index.html","6c4f6e7762c6bd6feaf04c04ec2d7e4b"],["E:/GitHubBlog/public/2021/01/17/OpenNMT源码解析/index.html","37aa3f5f93338c42ac5f5fa69d367bb5"],["E:/GitHubBlog/public/2021/01/19/重学数据结构——线性表/index.html","cdf3502f9c965c6b4e4869704ab37b8c"],["E:/GitHubBlog/public/2021/01/22/《深度学习入门-基于python的理论与实现》——第二章 感知机/index.html","b3bf2477b11be05380819cd9bc66fa3d"],["E:/GitHubBlog/public/2021/01/22/代码理论/index.html","dd88739d499ac8c234ea965f67b9ed36"],["E:/GitHubBlog/public/2021/01/26/title复现中的问题/index.html","7c328d58b1c3f339fea7cdb5768adbd6"],["E:/GitHubBlog/public/2021/01/26/简单链表题/index.html","1719e8f7a1d9d5cf0906268331c662b9"],["E:/GitHubBlog/public/2021/01/28/DeepKeyphraseGeneration/index.html","9ec164ba0c6dd55f458ff03f28d96282"],["E:/GitHubBlog/public/2021/02/01/title模型详解/index.html","f261b2370198920c4331cd81a5c54c45"],["E:/GitHubBlog/public/2021/02/01/中等链表题/index.html","7b0309822bd2c900c94d4a4a23d6222e"],["E:/GitHubBlog/public/archives/2020/01/index.html","2c8639b4941e8c2e5b2672644fbdde56"],["E:/GitHubBlog/public/archives/2020/02/index.html","4e97b3e4e58a8d5d220336f2f8e1b735"],["E:/GitHubBlog/public/archives/2020/03/index.html","298d9daf3b141f2e052f12f33d11bbb3"],["E:/GitHubBlog/public/archives/2020/04/index.html","5a4ebf7c9f2c39ebd7b8702978c69d43"],["E:/GitHubBlog/public/archives/2020/05/index.html","7355efd9b62a0feb42cf96d75bff8e90"],["E:/GitHubBlog/public/archives/2020/07/index.html","e1dce030980c0cad6cd3a73f674cabe2"],["E:/GitHubBlog/public/archives/2020/08/index.html","15ecb8060a3839caf627d0657b8e1495"],["E:/GitHubBlog/public/archives/2020/09/index.html","63f97558b6fcbc08128b4ea38f60dbdf"],["E:/GitHubBlog/public/archives/2020/10/index.html","ed176d3cb2fc027a90733e590715a7d4"],["E:/GitHubBlog/public/archives/2020/11/index.html","3d9f4ddd1e4eaf298174bdd664ba599c"],["E:/GitHubBlog/public/archives/2020/12/index.html","c43073a8ec6b8e00dcdb0a1fab418b04"],["E:/GitHubBlog/public/archives/2020/index.html","ba072df8f9efb63a631a8d97aa40b051"],["E:/GitHubBlog/public/archives/2020/page/2/index.html","7e0a90b2521bc22e6a3c06ed25be6b6f"],["E:/GitHubBlog/public/archives/2020/page/3/index.html","02722b539e204507a10c18035fc6beb1"],["E:/GitHubBlog/public/archives/2020/page/4/index.html","ed61a517295fa981e16ca01e6159161e"],["E:/GitHubBlog/public/archives/2020/page/5/index.html","c5da6bd7de2a366f6adf500d860ee8df"],["E:/GitHubBlog/public/archives/2021/01/index.html","ce507163ff5e3d254c0cceabc96eac2c"],["E:/GitHubBlog/public/archives/2021/01/page/2/index.html","ff29dd511603a2f8c7800b7de95770ab"],["E:/GitHubBlog/public/archives/2021/02/index.html","c3a0bca51f21229f0c3aea5ae5360e48"],["E:/GitHubBlog/public/archives/2021/index.html","c81cea54a438eba8973f01b1b44d6785"],["E:/GitHubBlog/public/archives/2021/page/2/index.html","633c22e8f4150a9e7593e52b341eac33"],["E:/GitHubBlog/public/archives/index.html","15bcf73c6516ca48de9e7efefb51af49"],["E:/GitHubBlog/public/archives/page/2/index.html","13088f20b818b3fb50918b2cfe7f79cf"],["E:/GitHubBlog/public/archives/page/3/index.html","128da29638fd360ebfc87ef3be60b683"],["E:/GitHubBlog/public/archives/page/4/index.html","4f4e5e75ce5cef31e7449ba5ae8681a0"],["E:/GitHubBlog/public/archives/page/5/index.html","2f830a7a7b3ed5e894cbfa2c75d4fe27"],["E:/GitHubBlog/public/archives/page/6/index.html","87656c92b2326611571cae2f22e5b692"],["E:/GitHubBlog/public/archives/page/7/index.html","754da7927fde6293653a069c56d3d259"],["E:/GitHubBlog/public/assets/css/APlayer.min.css","fbe994054426fadb2dff69d824c5c67a"],["E:/GitHubBlog/public/assets/js/APlayer.min.js","8f1017e7a73737e631ff95fa51e4e7d7"],["E:/GitHubBlog/public/assets/js/Meting.min.js","bfac0368480fd344282ec018d28f173d"],["E:/GitHubBlog/public/css/404.css","b1bb50e1fabe41adcec483f6427fb3aa"],["E:/GitHubBlog/public/css/index.css","2389364df340948a5b822422956c2c87"],["E:/GitHubBlog/public/css/var.css","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/img/algolia.svg","fd40b88ac5370a5353a50b8175c1f367"],["E:/GitHubBlog/public/img/avatar.png","6cc4a809d23e3d8946a299ae4ce4e4cd"],["E:/GitHubBlog/public/index.html","bcb5a6a56470cda8a80cd14e2e585c3c"],["E:/GitHubBlog/public/js/copy.js","45aae54bf2e43ac27ecc41eb5e0bacf7"],["E:/GitHubBlog/public/js/fancybox.js","f84d626654b9bbc05720952b3effe062"],["E:/GitHubBlog/public/js/fireworks.js","35933ce61c158ef9c33b5e089fe757ac"],["E:/GitHubBlog/public/js/head.js","347edd99f8e3921b45fa617e839d8182"],["E:/GitHubBlog/public/js/hexo-theme-melody.js","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/js/katex.js","d971ba8b8dee1120ef66744b89cfd2b1"],["E:/GitHubBlog/public/js/scroll.js","a12ad7e37b9e325f8da3d0523d3757c7"],["E:/GitHubBlog/public/js/search/algolia.js","53160985d32d6fd66d98aa0e05b081ac"],["E:/GitHubBlog/public/js/search/local-search.js","1565b508bd866ed6fbd724a3858af5d8"],["E:/GitHubBlog/public/js/sidebar.js","d24db780974e661198eeb2e45f20a28f"],["E:/GitHubBlog/public/js/third-party/anime.min.js","9b4bbe6deb700e1c3606eab732f5eea5"],["E:/GitHubBlog/public/js/third-party/canvas-ribbon.js","09c181544ddff1db701db02ac30453e0"],["E:/GitHubBlog/public/js/third-party/jquery.fancybox.min.js","3c9fa1c1199cd4f874d855ecb1641335"],["E:/GitHubBlog/public/js/third-party/jquery.min.js","c9f5aeeca3ad37bf2aa006139b935f0a"],["E:/GitHubBlog/public/js/third-party/reveal/head.min.js","aad121203010122e05f1766d92385214"],["E:/GitHubBlog/public/js/third-party/reveal/highlight.min.js","44594243bec43813a16371af8fe7e105"],["E:/GitHubBlog/public/js/third-party/reveal/markdown.min.js","7ec4cef5a7fe3f0bf0eb4dc6d7bca114"],["E:/GitHubBlog/public/js/third-party/reveal/marked.min.js","c2a88705e206d71dc21fdc4445349127"],["E:/GitHubBlog/public/js/third-party/reveal/math.min.js","0a278fee2e57c530ab07f7d2d9ea8d96"],["E:/GitHubBlog/public/js/third-party/reveal/notes.min.js","89a0dfae4d706f9c75b317f686c3aa14"],["E:/GitHubBlog/public/js/third-party/reveal/reveal.min.js","8988419d67efb5fe93e291a357e26ec9"],["E:/GitHubBlog/public/js/third-party/reveal/zoom.min.js","9791f96e63e7d534cba2b67d4bda0419"],["E:/GitHubBlog/public/js/third-party/velocity.min.js","64da069aba987ea0512cf610600a56d1"],["E:/GitHubBlog/public/js/third-party/velocity.ui.min.js","c8ca438424a080620f7b2f4ee4b0fff1"],["E:/GitHubBlog/public/js/transition.js","911db4268f0f6621073afcced9e1bfef"],["E:/GitHubBlog/public/js/utils.js","3ff3423d966a1c351e9867813b3f6d36"],["E:/GitHubBlog/public/page/2/index.html","5708bf8b0eb0305de26f4571914563f7"],["E:/GitHubBlog/public/page/3/index.html","d88b9dfbc765791aa26705416c1a219d"],["E:/GitHubBlog/public/page/4/index.html","c27dd1003ef25ab508c92e0b986c95df"],["E:/GitHubBlog/public/page/5/index.html","f46e4dfbf5772cf7cf9dce43ff1873b8"],["E:/GitHubBlog/public/page/6/index.html","98a7fdf8f48b6fbb30c13ce48c5eb1fb"],["E:/GitHubBlog/public/page/7/index.html","544d19ab8999b3f8189a360afbd1cc81"],["E:/GitHubBlog/public/tags/Android/index.html","853201ad4b425f2b5d795fedd8a707a5"],["E:/GitHubBlog/public/tags/NLP/index.html","0cbabb787a8c82bb5f4a32285f19fafe"],["E:/GitHubBlog/public/tags/NLP/page/2/index.html","460a64823cbde3ac5ee1a3f5fa432595"],["E:/GitHubBlog/public/tags/NLP/page/3/index.html","53d251b265a081fe2868db27ab0169eb"],["E:/GitHubBlog/public/tags/R/index.html","9029357843ba1196165cb7331530fb58"],["E:/GitHubBlog/public/tags/index.html","403fa49a886ab6ef8a580ab933ba06c6"],["E:/GitHubBlog/public/tags/java/index.html","9add46f90b724fa54f12c0c2ce3842a4"],["E:/GitHubBlog/public/tags/java/page/2/index.html","57ecceb8feb3b3e4f3b5246b201832fb"],["E:/GitHubBlog/public/tags/leetcode/index.html","e1610217ad28a3826993265d4e596360"],["E:/GitHubBlog/public/tags/python/index.html","5ff80a5c731280fcde31e09d6194b782"],["E:/GitHubBlog/public/tags/pytorch/index.html","dab0df72ba8fa17ca11a17228639103a"],["E:/GitHubBlog/public/tags/优化方法/index.html","91a46e8a1255e95148b35024f30b5443"],["E:/GitHubBlog/public/tags/总结/index.html","3a368f3e47bb75f927844abc23501d74"],["E:/GitHubBlog/public/tags/总结/page/2/index.html","6c2d85d0756b2a0287e9c105da414982"],["E:/GitHubBlog/public/tags/数据分析/index.html","dff0cf356e51c77e8bebdeda029a2f51"],["E:/GitHubBlog/public/tags/数据挖掘/index.html","054e6b9bc50e1c9695900e57d8ff4204"],["E:/GitHubBlog/public/tags/数据结构/index.html","2a875e85e6f11a186a3071307e5874d9"],["E:/GitHubBlog/public/tags/机器学习/index.html","d7d88feef651962b02403da896942423"],["E:/GitHubBlog/public/tags/深度学习/index.html","ff87e801a1aac24698af38238ee9797f"],["E:/GitHubBlog/public/tags/爬虫/index.html","f2827d6c67ff559f751cd3648618ddb0"],["E:/GitHubBlog/public/tags/笔记/index.html","9a240ec1310cb7b791b08b0c310f6ab4"],["E:/GitHubBlog/public/tags/论文/index.html","4bc66dcc45b88d0cc1905dc90093c202"],["E:/GitHubBlog/public/tags/论文/page/2/index.html","c3a6cdc383084d72bb4e1483d026ac20"],["E:/GitHubBlog/public/tags/论文/page/3/index.html","fd49472a55fe85912d680f6f24b2fadd"],["E:/GitHubBlog/public/tags/读书笔记/index.html","961cc8c4cbb58a15ae2996d99c183b03"]];
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







