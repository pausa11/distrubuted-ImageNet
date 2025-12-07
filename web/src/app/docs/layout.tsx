import DocsSidebar from "@/components/DocsSidebar";

export default function DocsLayout({
    children,
}: {
    children: React.ReactNode;
}) {
    return (
        <div className="flex min-h-[calc(100vh-4rem)]">
            <DocsSidebar />
            <div className="flex-1 p-8 md:p-12 overflow-x-hidden">
                <div className="max-w-4xl mx-auto">
                    {children}
                </div>
            </div>
        </div>
    );
}
