#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <netinet/ether.h>
#include <netinet/ip.h>
#include <netinet/tcp.h>
#include <netinet/udp.h>
#include <netinet/ip_icmp.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <linux/if_packet.h>
#include <sys/ioctl.h>
#include <net/if.h>
#include <time.h>
#include <termios.h>
#include <fcntl.h>
#include <errno.h>
#include <syslog.h>

// 用于统计数据包相关信息的结构体
typedef struct PacketStats {
    char startTime[30];
    char endTime[30];
    int macBroadcastCount;
    int macShortCount;
    int macLongCount;
    int macByteCount;
    int macPacketCount;
    int bitPerSecond;
    int macByteSpeed;
    int macPacketSpeed;
    int ipBroadcastCount;
    int ipByteCount;
    int ipPacketCount;
    int udpPacketCount;
    int icmpPacketCount;
    int icmpRedirectCount;
    int icmpDestinationCount;
} PacketStats;

// 创建用于捕获数据包的原始套接字
int create_socket() {
    int sockfd = socket(AF_PACKET, SOCK_RAW, htons(ETH_P_ALL));
    if (sockfd == -1) {
        syslog(LOG_ERR, "Socket creation failed: %s", strerror(errno));
        exit(EXIT_FAILURE);
    }
    return sockfd;
}

// 设置套接字接收缓冲区大小(可选,可根据实际情况调整)
void set_socket_buffer_size(int sockfd, int buffer_size) {
    if (setsockopt(sockfd, SOL_SOCKET, SO_RCVBUF, &buffer_size, sizeof(buffer_size)) == -1) {
        syslog(LOG_ERR, "Failed to set socket buffer size: %s", strerror(errno));
    }
}

// 将套接字绑定到指定的网络接口(这里假设接口名已知)
int bind_to_interface(int sockfd, const char *interface_name) {
    struct sockaddr_ll socket_address;
    memset(&socket_address, 0, sizeof(socket_address));
    socket_address.sll_family = AF_PACKET;

    // 获取接口索引
    struct ifreq ifr;
    memset(&ifr, 0, sizeof(ifr));
    strncpy(ifr.ifr_name, interface_name, IFNAMSIZ - 1);
    if (ioctl(sockfd, SIOCGIFINDEX, &ifr) == -1) {
        syslog(LOG_ERR, "Failed to get interface index: %s", strerror(errno));
        return -1;
    }

    socket_address.sll_ifindex = ifr.ifr_ifindex;

    // 绑定套接字到接口
    if (bind(sockfd, (struct sockaddr *)&socket_address, sizeof(socket_address)) == -1) {
    syslog(LOG_ERR, "Failed to bind socket to interface: %s", strerror(errno));
        return -1;
    }

    return 0;
}

// 底层模块主函数,初始化并准备好用于捕获数据包的套接字等资源
int init_packet_capture(const char *interface_name, int buffer_size) {
    int sockfd = create_socket();
    if (sockfd == -1) {
        return -1;
    }

    // 设置接收缓冲区大小(可根据实际调整合适的值)
    set_socket_buffer_size(sockfd, buffer_size);

    // 绑定到指定接口
    if (bind_to_interface(sockfd, interface_name) == -1) {
        close(sockfd);
        return -1;
    }

    return sockfd;
}

// 将 MAC 地址从二进制形式转换为字符串形式(辅助函数)
char *mac_ntoa(const unsigned char *mac) {
    static char mac_str[18];
    sprintf(mac_str, "%.2x:%.2x:%.2x:%.2x:%.2x:%.2x",
            mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);
    return mac_str;
}

// 将 IP 地址从二进制形式转换为字符串形式(辅助函数)
char *ip_ntoa(const struct in_addr *addr) {
    static char ip_str[INET_ADDRSTRLEN];
    inet_ntop(AF_INET, addr, ip_str, INET_ADDRSTRLEN);
    return ip_str;
}

// 中层模块:MAC 层处理相关函数示例
void print_ethernet(const struct ether_header *eth) {
    printf("Ethernet Header:\n");
    printf("Source MAC: %s\n", mac_ntoa(eth->ether_shost));
    printf("Destination MAC: %s\n", mac_ntoa(eth->ether_dhost));
    printf("Ethernet Type: 0x%04x\n", ntohs(eth->ether_type));
}

// 中层模块:IP 层处理相关函数示例
void print_ip(const struct iphdr *ip) {
    printf("IP Header:\n");
    char *src_ip = ip_ntoa((struct in_addr *)&ip->saddr);
    char *dst_ip = ip_ntoa((struct in_addr *)&ip->daddr);
    printf("Version: %d\n", ip->version);
    printf("Header Length: %d bytes\n", ip->ihl * 4);
    printf("Type of Service: 0x%02x\n", ip->tos);
    printf("Total Length: %d bytes\n", ntohs(ip->tot_len));
    printf("Identification: 0x%04x\n", ntohs(ip->id));
    printf("Flags: 0x%02x\n", ip->frag_off >> 13);
    printf("Fragment Offset: %d\n", (ip->frag_off & 0x1fff) * 8);
    printf("Time to Live: %d\n", ip->ttl);
    printf("Protocol: %d\n", ip->protocol);
    printf("Header Checksum: 0x%04x\n", ntohs(ip->check));
    printf("Source IP: %s\n", src_ip);
    printf("Destination IP: %s\n", dst_ip);
}

// 中层模块:TCP 处理相关函数示例
void print_tcp(const struct tcphdr *tcp) {
    printf("TCP Header:\n");
    printf("Source Port: %d\n", ntohs(tcp->source));
    printf("Destination Port: %d\n", ntohs(tcp->dest));
    printf("Sequence Number: %u\n", ntohs(tcp->seq));
    printf("Acknowledgment Number: %u\n", ntohs(tcp->ack_seq));
    printf("Data Offset: %d\n", tcp->doff * 4);
    printf("Reserved: 0x%04x\n", tcp->res1);
    printf("Flags:\n");
    printf(" URG: %d\n", (tcp->urg & 0x02) >> 1);
    printf(" ACK: %d\n", (tcp->ack & 0x10) >> 4);
    printf(" PSH: %d\n", (tcp->psh & 0x08) >> 3);
    printf(" RST: %d\n", (tcp->rst & 0x04) >> 2);
    printf(" SYN: %d\n", (tcp->syn & 0x02) >> 1);
    printf(" FIN: %d\n", tcp->fin & 0x01);
    printf("Window Size: %d\n", ntohs(tcp->window));
    printf("Checksum: 0x%04x\n", ntohs(tcp->check));
    printf("Urgent Pointer: %d\n", ntohs(tcp->urg_ptr));
}

// 中层模块:UDP 处理相关函数示例
void print_udp(const struct udphdr *udp) {
    printf("UDP Header:\n");
    printf("Source Port: %d\n", ntohs(udp->source));
    printf("Destination Port: %d\n", ntohs(udp->dest));
    printf("Length: %d\n", ntohs(udp->len));
    printf("Checksum: 0x%04x\n", ntohs(udp->check));
}

// 中层模块:ICMP 处理相关函数示例
void print_icmp(const struct icmphdr *icmp) {
    printf("ICMP Header:\n");
    printf("Type: %d\n", icmp->type);
    printf("Code: %d\n", icmp->code);
    printf("Checksum: 0x%04x\n", ntohs(icmp->checksum));

    // 检查 ICMP 类型是否为 ECHOREPLY
    if (icmp->type == ICMP_ECHOREPLY) {
        // 根据 icmp 结构体的定义正确访问 identifier 成员（这里应该是 id 成员）
        printf("Identifier: %d\n", ntohs(icmp->un.echo.id));
        printf("Sequence Number: %d\n", ntohs(icmp->un.echo.sequence));
    }
}

// 根据数据包内容分析并调用相应的协议处理函数(核心分析函数)
void analyze_packet(const void *packet, PacketStats *stats) {
    const struct ether_header *eth = (const struct ether_header *)packet;

    // 判断 MAC 地址类型并计数
    if (memcmp(eth->ether_dhost, "\xff\xff\xff\xff\xff\xff", 6) == 0) {
        stats->macBroadcastCount++;
    } else if ((eth->ether_dhost[0] & 1) == 1) {
        // 这里简单判断组播地址（不区分短和长，可根据实际需求细化）
        stats->macShortCount++;
    } else {
        stats->macLongCount++;
    }

    stats->macByteCount += sizeof(struct ether_header);
    stats->macPacketCount++;

    switch (ntohs(eth->ether_type)) {
        case ETHERTYPE_IP: {
            const struct iphdr *ip = (const struct iphdr *)(packet + sizeof(struct ether_header));

            // 判断 IP 广播地址并计数
            if (ip->daddr == 0xffffffff) {
                stats->ipBroadcastCount++;
            }

            stats->ipByteCount += ntohs(ip->tot_len);
            stats->ipPacketCount++;

            print_ip(ip);

            switch (ip->protocol) {
                case IPPROTO_TCP: {
                    const struct tcphdr *tcp = (const struct tcphdr *)(packet + sizeof(struct ether_header) + (ip->ihl * 4));
                    print_tcp(tcp);
                    break;
                }
                case IPPROTO_UDP: {
                    const struct udphdr *udp = (const struct udphdr *)(packet + sizeof(struct ether_header) + (ip->ihl * 4));
                    print_udp(udp);
                    stats->udpPacketCount++;
                    break;
                }
                case IPPROTO_ICMP: {
                    const struct icmphdr *icmp = (const struct icmphdr *)(packet + sizeof(struct ether_header) + (ip->ihl * 4));
                    print_icmp(icmp);
                    stats->icmpPacketCount++;
                    // 进一步判断 ICMP 类型并计数
                    if (icmp->type == ICMP_REDIRECT) {
                        stats->icmpRedirectCount++;
                    } else if (icmp->type == ICMP_DEST_UNREACH) {
                        stats->icmpDestinationCount++;
                    }
                    break;
                }
                default:
                    printf("Unsupported IP protocol: %d\n", ip->protocol);
                    break;
            }
            break;
        }
        case ETHERTYPE_ARP:
            // 这里可以添加对 ARP 协议的详细处理逻辑,目前简单打印提示
            printf("ARP packet received (not fully processed here)\n");
            break;
        default:
            printf("Unsupported Ethernet type: 0x%04x\n", ntohs(eth->ether_type));
            break;
    }
}

// 初始化数据包统计结构体
void init_packet_stats(PacketStats *stats) {
    time_t current_time = time(NULL);
    struct tm *time_info = localtime(&current_time);
    strftime(stats->startTime, sizeof(stats->startTime), "%H:%M:%S %b %d %Y", time_info);

    stats->macBroadcastCount = 0;
    stats->macShortCount = 0;
    stats->macLongCount = 0;  // 添加声明
    stats->macByteCount = 0;
    stats->macPacketCount = 0;
    stats->bitPerSecond = 0;
    stats->macByteSpeed = 0;
    stats->macPacketSpeed = 0;
    stats->ipBroadcastCount = 0;
    stats->ipByteCount = 0;
    stats->ipPacketCount = 0;
    stats->udpPacketCount = 0;
    stats->icmpPacketCount = 0;
    stats->icmpRedirectCount = 0;
    stats->icmpDestinationCount = 0;
}

// 获取当前时间并更新到统计结构体的结束时间字段
void update_end_time(PacketStats *stats) {
    time_t current_time = time(NULL);
    struct tm *time_info = localtime(&current_time);
    strftime(stats->endTime, sizeof(stats->endTime), "%H:%M:%S %b %d %Y", time_info);
}
// 检查是否有按键按下的函数
int kbhit() {
    struct termios oldt, newt;
    int ch;
    int oldf;

    tcgetattr(STDIN_FILENO, &oldt);
    newt = oldt;
    newt.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);
    oldf = fcntl(STDIN_FILENO, F_GETFL, 0);
    fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);

    ch = getchar();

    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
    fcntl(STDIN_FILENO, F_SETFL, oldf);

    if (ch!= EOF) {
        ungetc(ch, stdin);
        return 1;
    }

    return 0;
}
// 数据包协议统计模块,统计不同协议的数据包数量(按照详细要求实现)
void protocol_stats(PacketStats *stats) {
    time_t startTime = time(NULL);
    double totalElapsedTime = 0;
    time_t lastTime = startTime;

    printf("Protocol Statistics:\n");
    printf("StartTime: %s\n", stats->startTime);

    // 循环计算累计时间差和统计信息
    while (1) {
        time_t currentTime = time(NULL);
        double elapsedTime = difftime(currentTime, lastTime);
        totalElapsedTime += elapsedTime;
        lastTime = currentTime;

        // 检查是否有按键按下，若按下 'e' 则退出循环
        if (kbhit()) {
            char ch = getchar();
            if (ch == 'e') {
                break;
            }
        }
    }

    // 根据累计时间差计算速率
    if (totalElapsedTime > 0) {
        stats->bitPerSecond = (stats->macByteCount * 8) / totalElapsedTime;
        stats->macByteSpeed = stats->macByteCount / totalElapsedTime;
        stats->macPacketSpeed = stats->macPacketCount / totalElapsedTime;
    }

    update_end_time(stats);
    printf("EndTime: %s\n", stats->endTime);
    printf("MAC Broadcast: %d\n", stats->macBroadcastCount);
    printf("MAC Short: %d\n", stats->macShortCount);
    printf("MAC Long: %d\n", stats->macLongCount);
    printf("MAC Byte: %d\n", stats->macByteCount);
    printf("MAC Packet: %d\n", stats->macPacketCount);
    if (totalElapsedTime > 0) {
        printf("Bit/S: %d\n", stats->bitPerSecond);
        printf("MAC ByteSpeed: %d\n", stats->macByteSpeed);
        printf("MAC PacketSpeed: %d\n", stats->macPacketSpeed);
    } else {
        printf("Bit/S: N/A (Insufficient time elapsed)\n");
        printf("MAC ByteSpeed: N/A (Insufficient time elapsed)\n");
        printf("MAC PacketSpeed: N/A (Insufficient time elapsed)\n");
    }
    printf("IP Broadcast: %d\n", stats->ipBroadcastCount);
    printf("IP Byte: %d\n", stats->ipByteCount);
    printf("IP Packet: %d\n", stats->ipPacketCount);
    printf("UDP Packet: %d\n", stats->udpPacketCount);
    printf("ICMP Packet: %d\n", stats->icmpPacketCount);
    printf("ICMP Redirect: %d\n", stats->icmpRedirectCount);
    printf("ICMP Destination: %d\n\n\n", stats->icmpDestinationCount);
}




// 主函数,模拟整个数据包监控及统计流程(示例)
int main() {
    PacketStats stats;
    init_packet_stats(&stats);

    int sockfd = init_packet_capture("ens33", 65536);
    if (sockfd == -1) {
        return -1;
    }

    time_t lastPacketTime;  // 在此处声明 lastPacketTime

    while (1) {
        char packet_buffer[1500];
        ssize_t packet_size = recv(sockfd, packet_buffer, sizeof(packet_buffer), 0);
        if (packet_size == -1) {
            perror("Packet reception failed");
            continue;
        }
        analyze_packet(packet_buffer, &stats);
        // 更新统计信息
        stats.macByteCount += packet_size;
        stats.macPacketCount++;
        if (ntohs(((struct ether_header *)packet_buffer)->ether_type) == ETHERTYPE_IP) {
            const struct iphdr *ip = (const struct iphdr *)(packet_buffer + sizeof(struct ether_header));
            stats.ipByteCount += ntohs(ip->tot_len);
            stats.ipPacketCount++;
            if (ip->protocol == IPPROTO_UDP) {
                stats.udpPacketCount++;
            } else if (ip->protocol == IPPROTO_ICMP) {
                stats.icmpPacketCount++;
            }
        }

        // 更新时间
        time_t current_time = time(NULL);
        lastPacketTime = current_time;

        if (kbhit()) {
            char ch = getchar();
            if (ch == 'e') {
                break;
            }
        }
    }

    protocol_stats(&stats);
    close(sockfd);
    return 0;
}
