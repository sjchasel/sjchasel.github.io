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

var precacheConfig = [["E:/GitHubBlog/public/2020/01/28/hello-world/index.html","565bfbc490f126d3580afe2b7b69f7e2"],["E:/GitHubBlog/public/2020/02/08/python爬虫学习总结/index.html","a22aee7aee645bc5a284f2ec83216d43"],["E:/GitHubBlog/public/2020/02/19/数据结构第一周-绪论/index.html","7d9a9f62c85ecbe269e0b6f3f8f8babc"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-1/index.html","01de3ce7fddaf5918e14efe70ca035e8"],["E:/GitHubBlog/public/2020/02/27/《R语言实战》学习总结-2-创建数据集/index.html","4c5d725ec2cc4e44ee2fb3ca4e74333f"],["E:/GitHubBlog/public/2020/03/01/《R语言实战》学习总结-3-图形初阶/index.html","20dcff8387e556d186e0fbd3d83b2486"],["E:/GitHubBlog/public/2020/03/02/数据结构第二周-线性结构1/index.html","eec2c634c0c5d252785f90e194f40a78"],["E:/GitHubBlog/public/2020/03/03/《R语言实战》学习总结-4-基本数据管理/index.html","7b8d7ff3f819935304ebfccae8cc08bc"],["E:/GitHubBlog/public/2020/03/14/数据结构第三周-线性结构2/index.html","d8ec47d26282025a3e2ede20a026e6ef"],["E:/GitHubBlog/public/2020/03/14/数据结构第四周-栈与队列/index.html","e9d511476844e9946c085f1cec01fbb3"],["E:/GitHubBlog/public/2020/03/17/leetcode记录-1、88、20/index.html","2b8a479dab6a0fd1395818238e5d4f94"],["E:/GitHubBlog/public/2020/03/20/温度转换程序/index.html","82a54ff7d7768e7237383b38417f9eca"],["E:/GitHubBlog/public/2020/03/22/数据结构第五周-递归/index.html","186e412cb4c3e18d71e5b676ecf1dbcc"],["E:/GitHubBlog/public/2020/03/22/数据结构第六周-串与数组/index.html","68d9662281968643a2efb3b0cceecc62"],["E:/GitHubBlog/public/2020/04/02/leetcode记录-52/index.html","6c9451b406d8e228414112cc911546ae"],["E:/GitHubBlog/public/2020/04/17/Android第八周作业-每天第一次打开应用更新数据/index.html","21eadcc366e0aefb411fdf3af95450e0"],["E:/GitHubBlog/public/2020/05/21/attention模型/index.html","1af52c84b8920f52890beaa0d95b08ed"],["E:/GitHubBlog/public/2020/05/21/编码器与解码器框架/index.html","03ffb8a9e562fe685374020e646bcf6b"],["E:/GitHubBlog/public/2020/07/11/思考，快与慢/index.html","b951c0a4e3473995440b680cf67c7026"],["E:/GitHubBlog/public/2020/07/13/20200713-0717总结/index.html","d28747e18a4515a8eda5500a2ac392a4"],["E:/GitHubBlog/public/2020/07/13/ZeroShotCeres Zero-Shot Relation Extraction/index.html","d0af04d1f8d6ddc7e7fa7a1e08b62466"],["E:/GitHubBlog/public/2020/07/20/Keyphrase Generation A Text Summarization Struggle/index.html","e4e605bfa92d220518ab14e10a20619c"],["E:/GitHubBlog/public/2020/07/20/TPR/index.html","95d07cc3e2960c3734a5b3e8e6722e72"],["E:/GitHubBlog/public/2020/07/23/Keyphrase Generation for Scientific Document Retrieval/index.html","434385e8cdd293a6580b885a605da83e"],["E:/GitHubBlog/public/2020/07/24/20200718-0724总结/index.html","d598d5c11f4e4409a55c19f6191753d7"],["E:/GitHubBlog/public/2020/07/31/20200725-0731总结/index.html","84878bb4ed6bc92104b635e4f88ad01d"],["E:/GitHubBlog/public/2020/08/07/20200801-0807总结/index.html","387145d6987ca1de59fe6a13c678fe10"],["E:/GitHubBlog/public/2020/08/12/逻辑回归算法/index.html","feec076b223ec848e4a5b9586df63c17"],["E:/GitHubBlog/public/2020/08/13/李宏毅-机器学习21/index.html","ed1986bc37cf6b1118b67434a65f106f"],["E:/GitHubBlog/public/2020/08/14/20200808-0815总结/index.html","114198efc14cc59478efe78042bb2c2f"],["E:/GitHubBlog/public/2020/09/03/20200816-0903总结/index.html","e74418ce0d078a7cf1c55e12d35cbedb"],["E:/GitHubBlog/public/2020/09/16/20200914-0918总结/index.html","683d99910405fd94347c8b8a98899cba"],["E:/GitHubBlog/public/2020/09/25/20200919-0925总结/index.html","1062ec46f52a25c3f4c381440a3ab769"],["E:/GitHubBlog/public/2020/10/06/《Deep Keyphrase Generation》详细解析！/index.html","4e8ec7b931df3edd5598f6ca7c740642"],["E:/GitHubBlog/public/2020/10/16/20201008-1017总结/index.html","ba0b27ef6f754335ed77734463006d88"],["E:/GitHubBlog/public/2020/10/20/特征选择的几种方法(一)————Filter过滤法/index.html","785a26e1274c70ca01dc0afe9012f8ec"],["E:/GitHubBlog/public/2020/10/21/PCA/index.html","beb2a22b30b2f3fe6a04e900f34f2343"],["E:/GitHubBlog/public/2020/10/22/20201018-1023总结/index.html","248b28d6cd72f2c70ed2610dfcddc7ef"],["E:/GitHubBlog/public/2020/10/24/20201024-1030/index.html","2c58d771e5f5010e5e6869e8c646aad2"],["E:/GitHubBlog/public/2020/10/30/1030讨论记录/index.html","b8a37888416ecf9971ebfc49c7672efe"],["E:/GitHubBlog/public/2020/11/03/20201103-总结/index.html","2243bd699e427545197f0922ca28129b"],["E:/GitHubBlog/public/2020/11/05/线性回归与逻辑回归/index.html","162e9f0b8b81cbdade75002afaafb12d"],["E:/GitHubBlog/public/2020/11/07/决策树/index.html","94ce9936a2df33600114c2a8a10d7277"],["E:/GitHubBlog/public/2020/11/13/20201107-1113总结/index.html","6aea581522f1ec4fff102dfcb76d8c8b"],["E:/GitHubBlog/public/2020/11/15/20201114-119总结/index.html","c787b551b0306176180cea72dc058096"],["E:/GitHubBlog/public/2020/11/16/手推一个SVM/index.html","7a33673f55665d63c4edf1186edfe949"],["E:/GitHubBlog/public/2020/11/27/20201127左右的总结/index.html","d234e9d54ba396af63075e3e582b3016"],["E:/GitHubBlog/public/2020/12/23/第一个深度学习模型（pytorch）/index.html","03bd1b587031941dc846dd9748623ea2"],["E:/GitHubBlog/public/2021/01/02/鲍鱼最终版/index.html","81ae3af47dfd2d0d6bb5abe14e463450"],["E:/GitHubBlog/public/2021/01/08/GSA_NumericalExperiment/index.html","ca8cd04fff2209f52dab7fbfc9db57cd"],["E:/GitHubBlog/public/2021/01/16/20210115-01总结/index.html","1cefa32f538909fb596e9a6e6afaa631"],["E:/GitHubBlog/public/2021/01/17/20200117-0118总结——OpenNMT阅读（二）/index.html","ddf473f6e984ca0df59a1e07b778c037"],["E:/GitHubBlog/public/2021/01/17/OpenNMT源码解析/index.html","707a06b138dc769437f314fed1506ba3"],["E:/GitHubBlog/public/2021/01/19/重学数据结构——线性表/index.html","4a0bbd3b7cd09e6afa695897079a5b48"],["E:/GitHubBlog/public/2021/01/22/《深度学习入门-基于python的理论与实现》——第二章 感知机/index.html","694b7f09585a71bac7a534f30f64d8dc"],["E:/GitHubBlog/public/2021/01/22/代码理论/index.html","fca678820dfe6c47a14bbf6ccde743f3"],["E:/GitHubBlog/public/2021/01/26/title复现中的问题/index.html","574d25b93594c8bab3065aba79ac2259"],["E:/GitHubBlog/public/2021/01/26/简单链表题/index.html","f9dc27b34d2955fecec1ec992ff2f8bc"],["E:/GitHubBlog/public/2021/01/28/DeepKeyphraseGeneration/index.html","8a897c3caf72c5c05f4bef23f4bfc6a6"],["E:/GitHubBlog/public/2021/02/01/title模型详解/index.html","fbc54ffa2fef44c0e9e332a5c42e756a"],["E:/GitHubBlog/public/2021/02/14/中等链表题/index.html","3aa43086d22f03c5af0184744391d9ba"],["E:/GitHubBlog/public/2021/02/20/AcWing基础算法/index.html","fd623a9a80ef5413403aa55b326defc1"],["E:/GitHubBlog/public/2021/02/20/DCIC共享单车task01/index.html","b8c89a994a478a9d3dd543ef495e4855"],["E:/GitHubBlog/public/2021/02/21/Teacher forcing in RNN/index.html","940a5a1cd728d98ec1d52213a622832a"],["E:/GitHubBlog/public/2021/02/24/DCIC共享单车task02/index.html","3bebe089dcee934eb68b96daf8284600"],["E:/GitHubBlog/public/archives/2020/01/index.html","a09a11133f14b12e46640e5c3fc526c2"],["E:/GitHubBlog/public/archives/2020/02/index.html","6d4830228107173432649c09ddf09466"],["E:/GitHubBlog/public/archives/2020/03/index.html","07f05fc16264d26cc21215ece49bc24a"],["E:/GitHubBlog/public/archives/2020/04/index.html","6189f838d373223199232b9f20b17a2d"],["E:/GitHubBlog/public/archives/2020/05/index.html","31ad13a534d40dd0674c6c0c54f3f52f"],["E:/GitHubBlog/public/archives/2020/07/index.html","bea1c04c1c20b45efaca64ed36d97b9f"],["E:/GitHubBlog/public/archives/2020/08/index.html","b0819171a1d8d37ae0917d74325aabe8"],["E:/GitHubBlog/public/archives/2020/09/index.html","92aa95bd4e363264f16488f0186baa7a"],["E:/GitHubBlog/public/archives/2020/10/index.html","3345e3e60eb0afead62a665787c03108"],["E:/GitHubBlog/public/archives/2020/11/index.html","2b3d89f921294ae1d6f3ce5c7a20ee57"],["E:/GitHubBlog/public/archives/2020/12/index.html","77428ef3689c63d8410ca058d0764d20"],["E:/GitHubBlog/public/archives/2020/index.html","2d109d0cf65548c9a74db2c22d36c92b"],["E:/GitHubBlog/public/archives/2020/page/2/index.html","86ec6ce8fc8cfe9a79db9fae75a033e2"],["E:/GitHubBlog/public/archives/2020/page/3/index.html","dc5bfb8ee846cbe635d4716e877ad8cb"],["E:/GitHubBlog/public/archives/2020/page/4/index.html","ce9ef5a60e6da583da48967b610e4ba7"],["E:/GitHubBlog/public/archives/2020/page/5/index.html","9882e6a3132f265fa85d1021b53f6798"],["E:/GitHubBlog/public/archives/2021/01/index.html","fceda68922b0f3febf73e65ed7036eab"],["E:/GitHubBlog/public/archives/2021/01/page/2/index.html","f1f14b0301726efb10b241ddc11f61c3"],["E:/GitHubBlog/public/archives/2021/02/index.html","bfd362957e9c2369f234d5cb7e5a53c0"],["E:/GitHubBlog/public/archives/2021/index.html","b5e01d8b5c147506ce72298acd5fe85e"],["E:/GitHubBlog/public/archives/2021/page/2/index.html","bc5c17470bb374f6a71129b6da77abb2"],["E:/GitHubBlog/public/archives/index.html","9be21525e8461b0c537edf1c28fc98dc"],["E:/GitHubBlog/public/archives/page/2/index.html","a2a72e275207b8fae0ddb0e552d14056"],["E:/GitHubBlog/public/archives/page/3/index.html","3a520274ed146ea11df3edb43bc39598"],["E:/GitHubBlog/public/archives/page/4/index.html","a2f1b59dbf80ef83f8d58b20b2a85039"],["E:/GitHubBlog/public/archives/page/5/index.html","6a0235a3251212f722f57d624daad106"],["E:/GitHubBlog/public/archives/page/6/index.html","8384b5381ae88e307fb9044d683c37d3"],["E:/GitHubBlog/public/archives/page/7/index.html","a8e51a0378a25678a580c9c59a545374"],["E:/GitHubBlog/public/assets/css/APlayer.min.css","fbe994054426fadb2dff69d824c5c67a"],["E:/GitHubBlog/public/assets/js/APlayer.min.js","8f1017e7a73737e631ff95fa51e4e7d7"],["E:/GitHubBlog/public/assets/js/Meting.min.js","bfac0368480fd344282ec018d28f173d"],["E:/GitHubBlog/public/css/404.css","b1bb50e1fabe41adcec483f6427fb3aa"],["E:/GitHubBlog/public/css/index.css","2389364df340948a5b822422956c2c87"],["E:/GitHubBlog/public/css/var.css","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/img/algolia.svg","fd40b88ac5370a5353a50b8175c1f367"],["E:/GitHubBlog/public/img/avatar.png","6cc4a809d23e3d8946a299ae4ce4e4cd"],["E:/GitHubBlog/public/index.html","7cc574da20106a91e0fbdbba1decc449"],["E:/GitHubBlog/public/js/copy.js","45aae54bf2e43ac27ecc41eb5e0bacf7"],["E:/GitHubBlog/public/js/fancybox.js","f84d626654b9bbc05720952b3effe062"],["E:/GitHubBlog/public/js/fireworks.js","35933ce61c158ef9c33b5e089fe757ac"],["E:/GitHubBlog/public/js/head.js","347edd99f8e3921b45fa617e839d8182"],["E:/GitHubBlog/public/js/hexo-theme-melody.js","d41d8cd98f00b204e9800998ecf8427e"],["E:/GitHubBlog/public/js/katex.js","d971ba8b8dee1120ef66744b89cfd2b1"],["E:/GitHubBlog/public/js/scroll.js","a12ad7e37b9e325f8da3d0523d3757c7"],["E:/GitHubBlog/public/js/search/algolia.js","53160985d32d6fd66d98aa0e05b081ac"],["E:/GitHubBlog/public/js/search/local-search.js","1565b508bd866ed6fbd724a3858af5d8"],["E:/GitHubBlog/public/js/sidebar.js","d24db780974e661198eeb2e45f20a28f"],["E:/GitHubBlog/public/js/third-party/anime.min.js","9b4bbe6deb700e1c3606eab732f5eea5"],["E:/GitHubBlog/public/js/third-party/canvas-ribbon.js","09c181544ddff1db701db02ac30453e0"],["E:/GitHubBlog/public/js/third-party/jquery.fancybox.min.js","3c9fa1c1199cd4f874d855ecb1641335"],["E:/GitHubBlog/public/js/third-party/jquery.min.js","c9f5aeeca3ad37bf2aa006139b935f0a"],["E:/GitHubBlog/public/js/third-party/reveal/head.min.js","aad121203010122e05f1766d92385214"],["E:/GitHubBlog/public/js/third-party/reveal/highlight.min.js","44594243bec43813a16371af8fe7e105"],["E:/GitHubBlog/public/js/third-party/reveal/markdown.min.js","7ec4cef5a7fe3f0bf0eb4dc6d7bca114"],["E:/GitHubBlog/public/js/third-party/reveal/marked.min.js","c2a88705e206d71dc21fdc4445349127"],["E:/GitHubBlog/public/js/third-party/reveal/math.min.js","0a278fee2e57c530ab07f7d2d9ea8d96"],["E:/GitHubBlog/public/js/third-party/reveal/notes.min.js","89a0dfae4d706f9c75b317f686c3aa14"],["E:/GitHubBlog/public/js/third-party/reveal/reveal.min.js","8988419d67efb5fe93e291a357e26ec9"],["E:/GitHubBlog/public/js/third-party/reveal/zoom.min.js","9791f96e63e7d534cba2b67d4bda0419"],["E:/GitHubBlog/public/js/third-party/velocity.min.js","64da069aba987ea0512cf610600a56d1"],["E:/GitHubBlog/public/js/third-party/velocity.ui.min.js","c8ca438424a080620f7b2f4ee4b0fff1"],["E:/GitHubBlog/public/js/transition.js","911db4268f0f6621073afcced9e1bfef"],["E:/GitHubBlog/public/js/utils.js","3ff3423d966a1c351e9867813b3f6d36"],["E:/GitHubBlog/public/page/2/index.html","69fbef64ac731e96f1612ddf5cd82438"],["E:/GitHubBlog/public/page/3/index.html","1fa47567eca4f110daddfac576492fd5"],["E:/GitHubBlog/public/page/4/index.html","93c49a7684fdae419f1c0b82d63ebf2d"],["E:/GitHubBlog/public/page/5/index.html","d67d27aca47c3aec3c3ba81d69c1119c"],["E:/GitHubBlog/public/page/6/index.html","a98927e426294a59988126f0c68c5179"],["E:/GitHubBlog/public/page/7/index.html","706f79d08be6cb25d389a80f0292e94f"],["E:/GitHubBlog/public/tags/Android/index.html","b843d2061ea39ecdc1a51832ad828738"],["E:/GitHubBlog/public/tags/NLP/index.html","86291c61a3a25fd91577f88b1e7ca6b2"],["E:/GitHubBlog/public/tags/NLP/page/2/index.html","9a48a970e8bbcbfca72c34dcc2723a83"],["E:/GitHubBlog/public/tags/NLP/page/3/index.html","bf9c7fffcd668346ba9f65f0a7484518"],["E:/GitHubBlog/public/tags/R/index.html","e3a60a8da5566d0f81c954de8c41d9fc"],["E:/GitHubBlog/public/tags/index.html","481462aa7eb76a3bbafd849510f43e9c"],["E:/GitHubBlog/public/tags/java/index.html","ce5e2a8de1289e47ea3da83cf3cc82d1"],["E:/GitHubBlog/public/tags/java/page/2/index.html","90da4ad10ee5cc56874be900a317d593"],["E:/GitHubBlog/public/tags/leetcode/index.html","18a0950a373917268b1c89bbce1be894"],["E:/GitHubBlog/public/tags/python/index.html","f9e669f78795afd35c27f955aceb9230"],["E:/GitHubBlog/public/tags/pytorch/index.html","23deb197ffba30c3fab6e0dace192ec2"],["E:/GitHubBlog/public/tags/优化方法/index.html","133850ac1a261a28e5934644e383bb6c"],["E:/GitHubBlog/public/tags/总结/index.html","4c8258289e8e4f14f01afb110edc204a"],["E:/GitHubBlog/public/tags/总结/page/2/index.html","6a49f2336988a33caf844f1da1ea5764"],["E:/GitHubBlog/public/tags/数据分析/index.html","471c71c3bb6971a6df356f79946c83df"],["E:/GitHubBlog/public/tags/数据挖掘/index.html","b9b31b5caa520677d4d6dd44e89a0b9c"],["E:/GitHubBlog/public/tags/数据结构/index.html","a415e5c9963e3d1a6407bb4f0b04795b"],["E:/GitHubBlog/public/tags/机器学习/index.html","1f2562b1200538d413e43c7399bc4441"],["E:/GitHubBlog/public/tags/比赛/index.html","d0462f77443089347c4d8ec8e95c7b86"],["E:/GitHubBlog/public/tags/深度学习/index.html","cd0e8d48e3016b0f7ca5c6d00668f7bd"],["E:/GitHubBlog/public/tags/爬虫/index.html","1ad65bab992b60bcfd41c104f20632b5"],["E:/GitHubBlog/public/tags/笔记/index.html","ac79878f8dc9c24e356cf9c60cf5dc31"],["E:/GitHubBlog/public/tags/算法/index.html","67d88721ed9da36a8238e8d5eb6849ab"],["E:/GitHubBlog/public/tags/论文/index.html","603450ee1cf6f57d5d4a967dce934f13"],["E:/GitHubBlog/public/tags/论文/page/2/index.html","fa7651578510c3488d7b7b86c65bfeaa"],["E:/GitHubBlog/public/tags/论文/page/3/index.html","ecbeb30bafc6f44b16a1b33397b1aa95"],["E:/GitHubBlog/public/tags/读书笔记/index.html","44b15ad6495d12f115b740b826725bdf"]];
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







