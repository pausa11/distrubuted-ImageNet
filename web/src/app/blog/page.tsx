import Link from "next/link";

const BLOG_POSTS = [
    {
        id: 1,
        title: "Introducing Our New Project",
        excerpt: "We are excited to announce the release of our new distributed system...",
        date: "Nov 29, 2025",
        author: "Team",
        category: "Announcement"
    },
    {
        id: 2,
        title: "Deep Dive into Architecture",
        excerpt: "Understanding the core components of our distributed neural network training...",
        date: "Nov 25, 2025",
        author: "Daniel",
        category: "Engineering"
    },
    {
        id: 3,
        title: "Performance Benchmarks",
        excerpt: "Comparing our results against state-of-the-art implementations...",
        date: "Nov 20, 2025",
        author: "Research",
        category: "Performance"
    }
];

export default function BlogPage() {
    return (
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
            <div className="text-center mb-16">
                <h1 className="text-4xl font-bold text-gray-900 sm:text-5xl">Blog</h1>
                <p className="mt-4 text-xl text-gray-500">
                    Thoughts, stories, and ideas from the team.
                </p>
            </div>

            <div className="grid gap-8 md:grid-cols-2 lg:grid-cols-3">
                {BLOG_POSTS.map((post) => (
                    <article key={post.id} className="bg-white rounded-xl shadow-sm border border-gray-100 overflow-hidden hover:shadow-md transition-shadow">
                        <div className="p-6">
                            <div className="flex items-center justify-between mb-4">
                                <span className="px-3 py-1 bg-blue-100 text-blue-800 text-xs font-medium rounded-full">
                                    {post.category}
                                </span>
                                <span className="text-sm text-gray-500">{post.date}</span>
                            </div>
                            <h2 className="text-xl font-bold text-gray-900 mb-2">
                                <Link href={`/blog/${post.id}`} className="hover:text-blue-600 transition-colors">
                                    {post.title}
                                </Link>
                            </h2>
                            <p className="text-gray-600 mb-4">
                                {post.excerpt}
                            </p>
                            <div className="flex items-center text-sm text-gray-500">
                                <span>By {post.author}</span>
                            </div>
                        </div>
                    </article>
                ))}
            </div>
        </div>
    );
}
