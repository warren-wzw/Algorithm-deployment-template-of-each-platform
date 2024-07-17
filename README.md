# 不同平台的算法部署模版
本项目旨在提供不同平台的算法部署模板，包括海思、瑞芯微、比特大陆，这些模板可帮助您在各种硬件平台上轻松部署算法，并提供了基本的配置指南。

## 部署案例
| 硬件平台        | 链接                                         |
|----------------|----------------------------------------------|
| HI3519DV500         | [Yolov5](https://github.com/warren-wzw/Algorithm-deployment-template-of-each-platform/tree/master/Hisi/Hi3519_DV500) |

## 支持的平台包括
```
Algorthim_Deployment/
├── Bitmain
│   └── BM1684X
├── Hisi
│   ├── Hi3519_DV500
│   └── SS928
├── images
│   ├── Bitmain
│   ├── Hisi
│   └── RockChip
├── LICENSE
├── readme.md
└── Rockchip
    ├── RK3588
    └── RV1126
```

## 海思（Hisilicon）平台

| 硬件平台        | 链接                                         |
|----------------|----------------------------------------------|
| SS928          | [SS928部署教程](https://github.com/warren-wzw/Algorithm-deployment-template-of-each-platform/tree/master/Hisi/SS928) |
| Hi3519_DV500  | [Hi3519_DV500部署教程](https://github.com/warren-wzw/Algorithm-deployment-template-of-each-platform/tree/master/Hisi/Hi3519_DV500) |


部署步骤：

    准备环境：安装适用于海思平台的开发工具链和依赖库。
    导出模型：导出你的模型文件。
    转换模型文件：将训练好的模型转换为海思平台可用的格式。
    配置硬件参数：根据目标硬件平台的特性，调整算法参数和性能设置。
    测试和调优：在目标平台上测试部署的算法，并根据性能表现进行调优。

## 瑞芯微（Rockchip）平台
| 硬件平台        | 链接                                         |
|----------------|----------------------------------------------|
| RV1126          | [RV1126部署教程](https://github.com/warren-wzw/Algorithm-deployment-template-of-each-platform/tree/master/Rockchip/RV1126) |
| RK3588  | [RK3588部署教程](https://github.com/warren-wzw/Algorithm-deployment-template-of-each-platform/tree/master/Rockchip/RK3588) 

部署步骤：

    安装环境：配置适用于瑞芯微平台的开发环境和工具链。
    导出模型：导出你的模型文件。
    模型转换：将训练好的模型转换为Rockchip平台支持的格式。
    硬件优化：根据硬件平台的特性进行性能优化和参数调整。
    部署测试：在目标平台上进行算法的部署测试，并评估性能和稳定性。

## 比特大陆（Bitmain）平台
| 硬件平台        | 链接                                         |
|----------------|----------------------------------------------|
| BM1684X          | [BM1684X部署教程](https://github.com/warren-wzw/Algorithm-deployment-template-of-each-platform/tree/master/Bitmain/BM1684X) |

部署步骤：

    配置环境：安装比特大陆平台所需的开发工具和依赖库。
    导出模型：导出你的模型文件。
    模型转换：将训练好的模型转换为比特大陆平台支持的格式。
    参数调整：根据硬件规格和性能要求，调整算法参数和模型设置。
    性能评估：在比特大陆平台上进行性能评估和调优，确保算法在硬件上正常运行。

其他注意事项：

    每个平台可能有不同的编译工具链和模型转换工具，确保根据目标平台的要求进行相应的配置和调整。
    请参考各平台的官方文档和开发者社区，获取更多关于算法部署和性能优化的指导和建议。

## 问题反馈
以上是对不同平台算法部署模板的基本介绍和指南。根据具体的硬件平台和算法需求，您可能需要进一步深入研究和实践，以实现最佳的算法性能和效果。
 
## 关于作者
* warren@伟
* 个人博客：其他内容可以参考我的博客[CSDN-warren@伟](https://blog.csdn.net/warren103098?type=blog)

